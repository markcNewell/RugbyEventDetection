import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('./alphapo')

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter




class AlphaPose:
    """docstring for ClassName"""
    def __init__(self, args):
        self.cfg = update_config(args.cfg)

        args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        args.detbatch = args.detbatch * len(args.gpus)
        args.posebatch = args.posebatch * len(args.gpus)
        args.tracking = (args.detector == 'tracker')

        self.mode, self.input_source = self.check_input(args)

        # Load pose model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

        if len(args.gpus) > 1:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=args.gpus).to(args.device)
        else:
            self.pose_model.to(args.device)
        self.pose_model.eval()

        # Init data writer
        queueSize = args.qsize
        self.writer = DataWriter(self.cfg, args, save_video=False, queueSize=queueSize).start()
        self.args = args
        




    def check_input(self, args):
        # for images
        if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
            inputpath = args.inputpath
            inputlist = args.inputlist
            inputimg = args.inputimg

            if len(inputlist):
                im_names = open(inputlist, 'r').readlines()
            elif len(inputpath) and inputpath != '/':
                for root, dirs, files in os.walk(inputpath):
                    im_names = files
            elif len(inputimg):
                im_names = [inputimg]

            return 'image', im_names

        else:
            raise NotImplementedError


    def print_finish_info(self, args):
        print('===========================> Finish Model Running.')
        if (args.save_img or args.save_video) and not args.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


    def loop(self):
        n = 0
        while True:
            yield n
            n += 1


    def predict(self, image):
        args = self.args
        # Load detection loader
        det_loader = DetectionLoader(self.input_source, [image], get_detector(args), self.cfg, args, batchSize=args.detbatch, mode=self.mode).start()

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        batchSize = args.posebatch
        if args.flip:
            batchSize = int(batchSize / 2)



        try:
            for i in im_names_desc:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        self.writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                        continue
                    if args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(args.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        if args.flip:
                            inps_j = torch.cat((inps_j, flip(inps_j)))
                        hm_j = self.pose_model(inps_j)
                        if args.flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], det_loader.joint_pairs, shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    if args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))

                    if args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

                if args.profile:
                    # TQDM
                    im_names_desc.set_description(
                        'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                            dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                    )
            self.print_finish_info(args)
            while(self.writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(self.writer.count()) + ' images in the queue...')
            self.writer.stop()
            det_loader.stop()
        except KeyboardInterrupt:
            self.print_finish_info(args)
            # Thread won't be killed when press Ctrl+C
            if args.sp:
                det_loader.terminate()
                while(self.writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining ' + str(self.writer.count()) + ' images in the queue...')
                self.writer.stop()
            else:
                # subprocesses are killed, manually clear queues
                self.writer.commit()
                self.writer.clear_queues()
                # det_loader.clear_queues()
        final_result = self.writer.results()
        return write_json(final_result, args.outputpath, form=args.format, for_eval=args.eval)

