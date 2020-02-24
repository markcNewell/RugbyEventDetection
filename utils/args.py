class Args:
    def __init__(self, config, checkpoint, inputimg):
        self.cfg = config
        self.checkpoint = checkpoint
        self.sp = True
        self.detector = "yolo"
        self.inputpath = ""
        self.inputlist = ""
        self.inputimg = inputimg
        self.outputpath = "examples/res/"
        self.save_img = True
        self.vis = False
        self.profile = False
        self.format = ""
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