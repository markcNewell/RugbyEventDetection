#LOCAL
from utils import config, util
from segmentation import segmentation
from clusters import clusters
from classification import classifier
from pose_estimation import preprocessor
from alphapo.scripts.alphapose import AlphaPose
from alphapo.args import Args


#PACKAGES
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import json
import datetime
import pickle
import shutil


def main():
	CONFIG_FILE = "./config/config.yaml"

	#Load config
	print("Loading configuration...", end="")
	args = config.get_parser(CONFIG_FILE)
	print("Done")


	#Initialise the segmentation predictor once to hold model for all predictions
	print("Loading segmentation model...", end="")
	predictor = segmentation.SegmentationPredictor(args)
	print("Done")


	#Initialise the pose estimator
	print("Loading pose model...", end="")
	ap = AlphaPose(Args(CONFIG_FILE, args.POSE_MODEL))
	print("Done")


	#Initialise classification model
	print("Loading classification model...", end="")
	nn_classifier = pickle.load(open(args.CLASSIFICATION_MODEL, 'rb'))
	print("Done")


	#Load in all the images from the in_dir folder
	if (args.IMAGE_INPUT) & (not args.VIDEO_INPUT):

		images = util.get_images(args.IN_DIR)

	elif (args.VIDEO_INPUT) & (not args.IMAGE_INPUT):

		#If working on a video convert the input video into a series of images
		#Skipping certain frames in order to make it the correct framerate
		#Given the option to reduce framerate as can be slow to process many frames
		print("Converting video to input frames...", end="")
		images = util.video_to_frames(args.IN_DIR, args.VIDEO_FILE, args.FRAMERATE)
		args.IN_DIR = os.path.join(args.IN_DIR, "frames")
		print("Done")

	else:
		raise ValueError("VIDEO input and IMAGE input can't happen simultaneously")


	#Tags object to store data calculated
	tags = {}


	#For each image
	for i, image_path in enumerate(images):

		start = datetime.datetime.now()


		#Debugging
		util.print_progress_bar(i,len(images),suffix="{}/{}".format(i,len(images)))


		#Setup paths
		inpath = os.path.join(args.IN_DIR, image_path)
		outpath = os.path.join(args.OUT_DIR, image_path)


		#Load Image
		image = cv2.imread(inpath, cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


		#Predict mask
		mask = predictor.predict(image)


		#Convert mask from PIL format to numpy/opencv
		mask = np.array(mask) * 255
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
		mask = cv2.bitwise_not(mask)


		#Get clusters
		out = clusters.makemask(image,mask)
		
		image_clusters, dimentions = clusters.extractclusters(out,image,bounding=True)
		plt.imsave("hey.png", image_clusters[0])


		if len(image_clusters) > 0:
			#Just work with single biggest cluster


			if 0 in image_clusters[0].shape:
				continue


			#Save the cluster for visual debugging
			#plt.imsave(os.path.join('cluster_imgs',image_path),image_clusters[0])


			#Run through alphapose to get json of poses
			json_data = ap.predict(image_clusters[0], image_path)
			json_data = json.loads(json_data)


			#Classify
			try:
				cluster = nn_classifier.predict(json_data)
			except:
				continue


			#Convert back to textual format
			tag = nn_classifier.le.inverse_transform(cluster)


			#Unpack dimentions of cluster
			x,y,w,h = dimentions


			#Draw bounding box and add tag annotation to original image
			cv2.rectangle(image, (x, y), (x+w, y+h), args.FONT_COLOR, args.FONT_THICKNESS)
			cv2.putText(image, tag[0], ((x+w+10),(y-10)), args.FONT, args.FONT_SCALE, args.FONT_COLOR, args.FONT_THICKNESS)


			#Save original image with bounding box and associated tag
			plt.imsave(outpath, image)


			#Update the ouput object with image tag
			tags[image_path] = {"tag": tag[0]}


	end = datetime.datetime.now()



	#Print once again to show 100%
	util.print_progress_bar(len(images),len(images))


	#Output the results object
	with open(os.path.join(args.OUT_DIR,"results.json"), "w") as file:
		json.dump(tags, file)
		


	if args.EVALUATE:
		acc = 0
		data = preprocessor.import_json(args.TEST_DATASET)

		for image in images:
			try:
				if data[image]['tag'] == tags[image]['tag']:
					acc += 1
			except:
				continue;

		acc = acc/len(images)

		print("Acuracy on test dataset:", acc)


	print("Time took:", end-start)


	#If video input delete frames created and make new video from outputed frames
	if args.VIDEO_INPUT:

		#Delete frames folder
		shutil.rmtree(args.IN_DIR, ignore_errors=False, onerror=None)


		#make video
		img_array = []
		for filename in glob.glob(args.OUT_DIR + '/*.png'):
		    img = cv2.imread(filename)
		    height, width, layers = img.shape
		    size = (width,height)
		    img_array.append(img)

		out = cv2.VideoWriter(os.path.join(args.OUT_DIR, 'output.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
		for i in range(len(img_array)):
		    out.write(img_array[i])
		out.release()





if __name__ == '__main__':
	main()
