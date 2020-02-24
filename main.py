#LOCAL
from utils import config, util
from segmentation import segmentation
from clusters import clusters
from classification import classifier
from pose_estimation import preprocessor
from alphapose.scripts import demo_inference as di
from alphapose.args import Args

#REQUIREMENTS
import matplotlib.pyplot as plt
import cv2
import os
import random



def main():
	#Load config - TODO: add AlphaPose config to same file
	print("Loading configuration...", end="")
	args = config.get_parser("./config/config.yaml")
	print("Done")


	#Initialise the segmentation predictor once to hold model for all predictions
	print("Loading segmentation model...", end="")
	predictor = segmentation.SegmentationPredictor(args)
	print("Done")


	#Initialise classification model
	print("Loading classification model...", end="")
	nn_classifier = classifier.Neural_Network("./dataset/train/annotations/train.json")
	print("Done")


	# Environmental variables - TODO: move to cmd command or to config file
	in_dir = "dataset/train/images/"
	out_dir = "out"
	poses = []


	#Load in all the images from the in_dir folder
	images = util.get_images(in_dir)


	#For each image
	for i, image_path in enumerate(images):


		#Debugging
		util.print_progress_bar(i,len(images),suffix="{}/{}".format(i,len(images)))


		#Setup paths
		inpath = os.path.join(in_dir, image_path)
		outpath = os.path.join(out_dir, image_path)


		#Load Image
		image = cv2.imread(inpath, cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


		#Predict mask
		mask = predictor.predict(image)
		mask.save(outpath)


		#Reload mask in correct format
		mask = cv2.imread(outpath, cv2.IMREAD_COLOR)


		#Get clusters
		out = clusters.makemask(image,mask)
		image_clusters = clusters.extractclusters(out,image)


		if len(image_clusters) > 0:
			#Just work with single biggest cluster for now


			if 0 in image_clusters[0].shape:
				continue


			#Save the cluster for visual debugging
			plt.imsave(os.path.join('cluster_imgs',image_path),image_clusters[0])


			#Run through alphapose to get json of poses
			json = di.main(Args("./configs/config.yaml", "./pretrained_models/fast_421_res152_256x192.pth"))


			#Classify
			cluster = nn_classifier.predict(json)


			#Convert back to textual format
			tag = nn_classifier.le.inverse_transform(cluster)


			print(tag)


	#Print once again to show 100%
	util.print_progress_bar(i+1,len(images))




if __name__ == '__main__':
	main()
