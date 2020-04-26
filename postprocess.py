from utils import util, config
import cv2, os, json
import argparse




def import_json(file):
	if os.path.exists(file):
		with open(file) as f:
			data = json.load(f)
			return data
	else:
		raise FileNotFoundException("No such json file")




def main(json_file, video_file, framerate, config_loc):
	CONFIG_FILE = config_loc

	#Load config
	print("Loading configuration...", end="")
	args = config.get_parser(CONFIG_FILE)
	print("Done")


	data = import_json(json_file)
	keys = list(data.keys())

	images = util.video_to_frames("", video_file, framerate)
	images, filenames = zip(*images)

	for i, filename in enumerate(filenames):
		if filename in keys:

			tag = data[filename]['tag']
			prob = data[filename]['prob']
			x = data[filename]['bbox']['x']
			y = data[filename]['bbox']['y']
			w = data[filename]['bbox']['w']
			h = data[filename]['bbox']['h']

			#Draw bounding box and add tag annotation to original image
			cv2.rectangle(images[i], (x, y), (x+w, y+h), args.FONT_COLOR, args.FONT_THICKNESS)
			cv2.putText(images[i], "{0}: {1:.3f}".format(tag,prob), ((x+w+10),(y-10)), args.FONT, args.FONT_SCALE, args.FONT_COLOR, args.FONT_THICKNESS)



	out = cv2.VideoWriter(os.path.join("", 'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), args.FRAMERATE, (images[0].shape[1], images[0].shape[0]))


	for i in range(len(images)):
	    out.write(images[i])
	out.release()

	print("Done")




if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--json', dest='jsonfile', help='The json file produced by in the results')
	parser.add_argument('--video', dest='videofile', help='The video file produced to produce the results')
	parser.add_argument('--framerate', dest='framerate', default=10, help='The framerate the results were produced at')
	parser.add_argument('--config', dest='config', default='./config/config.yaml', help='The config to load')


	args = parser.parse_args()
	main(args.jsonfile, args.videofile, args.framerate, args.config)


	
