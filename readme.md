# Trivial Event Detection in Commertial Rugby Footage #
A dissertation project by Mark Newell to detect events such as rucks, mauls and scrums from commertial rugby footage using computer vision techniques. Uses parts from:
1. Semseg - https://github.com/hszhao/semseg
2. AlphaPose - https://github.com/MVIG-SJTU/AlphaPose

## Prerequisits ##
1. Python 3.6.4
2. CUDA 10

## Setup ##
```
#Setup Environment
git clone https://github.com/markcNewell/RugbyEventDetection.git
cd RugbyEventDetection
pip install -r requirements.txt
mkdir "./out"
mkdir "./trained"
mkdir "./trained/pspnet"
git clone https://github.com/MVIG-SJTU/AlphaPose.git
mv AlphaPose alphapo
cd alphapo/

#Install alphapose - this compiling may take some time
export PATH=/usr/local/cuda/bin/:$PATH
!export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
!sudo apt-get install libyaml-dev
!python setup.py build develop --user
```
   
Once the packages have all built, the pretrained models need to be downloaded from the [model storage](https://drive.google.com/open?id=1UDiy7WQNvZpQAh2sgWgI-o2B9eqiL7pW) 
   
Put the `train_epoch_100.pth` file in `RugbyEventDetection/trained/pspnet/`
Unzip the dataset `dataset.zip` to `RugbyEventDetection`
Put the `yolov3-spp.weights` file in `RugbyEventDetection/alphapo/detector/yolo/data/`
Put the `fast_421_res152_256x192.pth` in `RugbyEventDetection/alphapo/pretrained_models/`
