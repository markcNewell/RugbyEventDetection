#Local Imports
from utils.util import colorize
from model.pspnet import PSPNet

#Packages
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

#Built-In
import os
import numpy as np
import cv2

class SegmentationPredictor(object):
  """
  A class to containerize functions to do with predicting the segmentation mask.
  """

  def __init__(self, args):
    self.args = args
    self.model = self.__setupSegmentation()



  def __setupSegmentation(self):
    if torch.cuda.device_count() == 0:
      device = torch.device("cpu")
    else:
      device = torch.device("cuda")



    model = PSPNet(layers=self.args.layers, classes=self.args.classes, zoom_factor=self.args.zoom_factor, pretrained=False)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    if os.path.isfile(self.args.model_path):
      checkpoint = torch.load(self.args.model_path, map_location=torch.device('cpu'))
      model.load_state_dict(checkpoint['state_dict'], strict=False)
      return model
    else:
      raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.model_path))



  def predict(self, image):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    colors = np.array([[255,255,255], [0,0,0]],dtype='uint8')

    result = self.__run(self.model.eval(), image, self.args.classes, mean, std,\
    self.args.base_size, self.args.test_h, self.args.test_w,\
    self.args.scales, colors)
    return result




  def __net_process(self, model, image, mean, std=None, flip=False):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()

    if std is None:
      for t, m in zip(input, mean):
        t.sub_(m)
    else:
      for t, m, s in zip(input, mean, std):
        t.sub_(m).div_(s)

    input = input.unsqueeze(0)

    if flip:
      input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
      output = model(input)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape

    if (h_o != h_i) or (w_o != w_i):
      output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)

    output = F.softmax(output, dim=1)

    if flip:
      output = (output[0] + output[1].flip(2)) / 2
    else:
      output = output[0]

    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output




  def __scale_process(self, model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    if pad_h > 0 or pad_w > 0:
      image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)

    for index_h in range(0, grid_h):
      for index_w in range(0, grid_w):
        s_h = index_h * stride_h
        e_h = min(s_h + crop_h, new_h)
        s_h = e_h - crop_h
        s_w = index_w * stride_w
        e_w = min(s_w + crop_w, new_w)
        s_w = e_w - crop_w
        image_crop = image[s_h:e_h, s_w:e_w].copy()
        count_crop[s_h:e_h, s_w:e_w] += 1
        prediction_crop[s_h:e_h, s_w:e_w, :] += self.__net_process(model, image_crop, mean, std)
        
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction





  def __run(self,model, image, classes, mean, std, base_size, crop_h, crop_w, scales, colors):
    h, w, _ = image.shape
    prediction = np.zeros((h, w, classes), dtype=float)

    for scale in scales:
      long_size = round(scale * base_size)
      new_h = long_size
      new_w = long_size
      if h > w:
        new_w = round(long_size/float(h)*w)
      else:
        new_h = round(long_size/float(w)*h)
        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prediction += self.__scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
      
    prediction = self.__scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    prediction = np.argmax(prediction, axis=2)
    gray = np.uint8(prediction)
    color = colorize(gray, colors)

    return color