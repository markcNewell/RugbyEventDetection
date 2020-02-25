mkdir "./RugbyEventDetection/out/"
mkdir "./RugbyEventDetection/trained"
mkdir "./RugbyEventDetection/trained/pspnet"
cp "./drive/My Drive/dissertation/train_epoch_100.pth" "./RugbyEventDetection/trained/pspnet/"

unzip "./drive/My Drive/dissertation/dataset.zip" -d "./RugbyEventDetection/"

cd RugbyEventDetection/

git clone https://github.com/MVIG-SJTU/AlphaPose.git

mv AlphaPose alphapo

cd alphapo/


# 3. install
!export PATH=/usr/local/cuda/bin/:$PATH
!export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
!sudo apt-get install libyaml-dev
!python setup.py build develop --user


!mkdir "./detector/yolo/data/"
!mkdir "./input/"
!cp "../../drive/My Drive/Models/yolov3-spp.weights" "./detector/yolo/data/"
!cp "../../drive/My Drive/Models/fast_421_res152_256x192.pth" "./pretrained_models/"
!cp "../../drive/My Drive/Models/fast_dcn_res50_256x192.pth" "./pretrained_models/"
