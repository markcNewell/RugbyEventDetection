#LOCAL
from utils import config, util
from segmentation import segmentation
from clusters import clusters
from classification import classifier
from pose_estimation import preprocessor
from alphapo.scripts import demo_inference as di
from alphapo.args import Args

#REQUIREMENTS
import matplotlib.pyplot as plt
import cv2
import os
import random
import json



FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
FONTCOLOR = (255,0,0)
FONTTHICKNESS = 2



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
	out_dir = "masks"
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
		image_clusters, dimentions = clusters.extractclusters(out,image)


		if len(image_clusters) > 0:
			#Just work with single biggest cluster for now


			if 0 in image_clusters[0].shape:
				continue


			#Save the cluster for visual debugging
			#plt.imsave(os.path.join('cluster_imgs',image_path),image_clusters[0])


			#Run through alphapose to get json of poses
			json_data = di.main(Args("./alphapo/configs/config.yaml", "./alphapo/pretrained_models/fast_421_res152_256x192.pth"), image_clusters[0])
			json_data = json.loads(json_data)


			#Classify
			cluster = nn_classifier.predict(json_data)
			#cluster = nn_classifier.clf.predict([[random.randint(0,400),random.randint(0,4)]]) #TESTING


			#Convert back to textual format
			tag = nn_classifier.le.inverse_transform(cluster)


			#Unpack dimentions of cluster
			x,y,w,h = dimentions


			#Draw bounding box and add tag annotation to original image
			cv2.rectangle(image, (x, y), (x+w, y+h), FONTCOLOR, FONTTHICKNESS)
			cv2.putText(image, tag[0], ((x+w+10),(y-10)), FONT, FONTSCALE, FONTCOLOR, FONTTHICKNESS)

			plt.imsave(os.path.join('output', image_path), image)



	#Print once again to show 100%
	util.print_progress_bar(i+1,len(images))




if __name__ == '__main__':
	main()
