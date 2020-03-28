import os, json, math, statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def import_json(file):
	if os.path.exists(file):
		with open(file) as f:
			data = json.load(f)
			return data
	else:
		raise FileNotFoundException("No such json file")




def get_file_names(data):
	return list(data.keys())




def index(i):
	return i * 3




def calculate_poses(data, files):
	all_poses = []
	for file in files:
		poses = []
		people = data[file]['people']
		for person in people:
			pose = calculate_pose(person['pose_keypoints_2d'])
			if (pose != 0):
				poses.append(pose)

		if len(poses) > 0:
			all_poses.append(statistics.mean(poses))



	return all_poses




def calculate_pose(keypoints, threshold):

	neck_index = 1
	l_hip_index = 8
	l_knee_index = 9
	r_hip_index = 11
	r_knee_index = 12

	neck = (keypoints[index(neck_index)],\
		keypoints[index(neck_index) + 1],\
		keypoints[index(neck_index) + 2])


	l_hip = (keypoints[index(l_hip_index)],\
		keypoints[index(l_hip_index) + 1],\
		keypoints[index(l_hip_index) + 2])


	l_knee = (keypoints[index(l_knee_index)],\
		keypoints[index(l_knee_index) + 1],\
		keypoints[index(l_knee_index) + 2])


	r_hip = (keypoints[index(r_hip_index)],\
		keypoints[index(r_hip_index) + 1],\
		keypoints[index(r_hip_index) + 2])


	r_knee = (keypoints[index(r_knee_index)],\
		keypoints[index(r_knee_index) + 1],\
		keypoints[index(r_knee_index) + 2])


	if (neck[2] < threshold) | (l_hip[2] < threshold) | (l_knee[2] < threshold) | (r_hip[2] < threshold) | (r_knee[2] < threshold):
		return 0
	else:
		pairs = get_key_pairs(neck,l_hip,l_knee,r_hip,r_knee)
		angle = calculate_angle(pairs)
		return angle




def get_key_pairs(neck,l_hip,l_knee,r_hip,r_knee):
	return (([neck,l_hip],[l_hip,l_knee]), ([neck,r_hip],[r_hip, r_knee]))




def calculate_angle(pairs):
	pose = []
	for side in pairs:
		angle = 0
		for pair in side:
			o = math.sqrt((pair[0][1] - pair[1][1])**2)
			a = math.sqrt((pair[0][0] - pair[1][0])**2)

			if (o == 0) & (a == 0) :
				raise ValueError("Keypoints in same place")
			elif a == 0:
				angle += 90
			elif o == 0:
				angle = 0
			else:
				angle += math.degrees(math.atan(o/a))

		pose.append(angle)

	pose = statistics.mean(pose)


	return pose  # The greater the angle the more upright the person




def tag_to_color(tags):
	colors = []
	for tag in tags:
		if tag == 'maul':
			colors.append([0,0,255])
		elif tag == 'ruck':
			colors.append([255,0,0])
		else:
			colors.append([0,255,0])
	return np.array(colors)





def get_attr(data, keys, a):
	attr = []

	for key in keys:
		attr.append(data[key][a])

	return attr



def get_indexes_for_tag(files, tags, t):
	indexes = []

	for i,tag in enumerate(tags):
		if tag == t:
			indexes.append(i)

	return indexes





def get_angles_and_ratios(filename, threshold):
	data = import_json(filename)
	files = get_file_names(data)
	poses = calculate_poses(data, files, threshold)

	ratios = get_attr(data, files, 'ratio')

	return poses, ratios





if __name__ == '__main__':
	data = import_json("../dataset/train/annotations/train.json")
	files = get_file_names(data)
	poses = calculate_poses(data, files)

	#Maul = blue (255,0,0)
	#ruck = red (0,0,255)
	#scrum = green (0,255,0)

	ratios = get_attr(data, files, 'ratio')

	tags = get_attr(data, files, 'tag')
	colours = tag_to_color(tags)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(ratios,poses,c=colours/255.0)


	fig.savefig('temp_2.png')


	LABELS = False


	if LABELS:

		indexes = get_indexes_for_tag(files, tags, 'maul')

		for i in indexes:		
			ax.annotate(files[i], (ratios[i], poses[i]))

		fig.savefig('temp_labeled.png')
	

