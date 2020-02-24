from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'alphapo/detector/yolo/cfg/yolov3-spp.cfg'
cfg.WEIGHTS = 'alphapo/detector/yolo/data/yolov3-spp.weights'
cfg.INP_DIM =  608
cfg.NMS_THRES =  0.6
cfg.CONFIDENCE = 0.05
cfg.NUM_CLASSES = 80