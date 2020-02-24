class Args:
    def __init__(self, config, checkpoint):
        self.cfg = config
        self.checkpoint = checkpoint
        self.sp = True
        self.detector = "yolo"
        self.inputpath = "../cluster_imgs/"
        self.inputlist = ""
        self.inputimg = ""
        self.outputpath = "examples/res/"
        self.save_img = True
        self.vis = False
        self.profile = False
        self.format = "open"
        self.min_box_area = 0
        self.detbatch = 5
        self.posebatch = 80
        self.eval = False
        self.gpus = "0"
        self.qsize = 1024
        self.flip = False
        self.debug = False
        self.video = ""
        self.webcam = 1
        self.save_video = False
        self.vis_fast = False
        self.pose_track = False