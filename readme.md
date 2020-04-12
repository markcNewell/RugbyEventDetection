# Trivial Event Detection in Commertial Rugby Footage #
A dissertation project by Mark Newell to detect events such as rucks, mauls and scrums from commertial rugby footage using computer vision techniques. Uses parts from:
1. Semseg - https://github.com/hszhao/semseg
2. AlphaPose - https://github.com/MVIG-SJTU/AlphaPose
3. Yolov3 - https://github.com/pjreddie/darknet

## Prerequisits ##
1. Python 3.6.4
2. CUDA 10

## Setup ##
The following setup is for a local environment. If you do however wish to run this in google colab (where most of development and testing took place) the .ipynb file has been provided.

### Setup Environment ###
```
git clone https://github.com/markcNewell/RugbyEventDetection.git
cd RugbyEventDetection
pip install -r requirements.txt
mkdir "./out"
mkdir "./trained"
mkdir "./trained/pspnet"
mkdir "./trained/classification"
```

### Install AlphaPose ###
```
git clone https://github.com/markcNewell/AlphaPose.git
mv AlphaPose alphapo
cd alphapo/
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
sudo apt-get install libyaml-dev
python setup.py build develop --user
```

### Install Darknet ###
```
git clone https://github.com/markcNewell/darknet.git
cd darknet
make
```

   
Once the packages have all built, the pretrained models need to be downloaded from the [model storage](https://drive.google.com/open?id=1UDiy7WQNvZpQAh2sgWgI-o2B9eqiL7pW):
   
* Put the `train_epoch_100.pth` file in `RugbyEventDetection/trained/pspnet/`
* Unzip the dataset `dataset.zip` to `RugbyEventDetection`
* Put the `yolov3-spp.weights` file in `RugbyEventDetection/alphapo/detector/yolo/data/`
* Put the `fast_421_res152_256x192.pth` in `RugbyEventDetection/alphapo/pretrained_models/`
* Put the `model.cfg` file in `RugbyEventDetection/darknet/model/`
* Put the `model.weights` file in `RugbyEventDetection/darknet/model/`


### Getting Started ###
A runner.sh script has been provided to showcase the running of the algorithm but it can be done by using the following commands:
* `python training.py --config config/training.yaml`
* `python main.py --config config/config-seg.yaml`

As the processing takes a number of minutes for a short video an example has been provided () the results.json and video file should be downloaded. A script has been provided to combine the two. `postprocessor.py` should be run with the flags `--json` and `--video` pointing to the respecive file locations.
