import json, argparse, statistics, preprocessor, math
import numpy as np


def index(i):
	return i * 3


def calculate_poses(data, files, threshold):
	all_poses = []
	for file in files:
		poses = []
		people = data[file]['people']
		for person in people:
			pose = calculate_pose(person['pose_keypoints_2d'], threshold)
			if (pose != 0):
				poses.append(pose)

		if len(poses) > 0:
			all_poses.append((file, statistics.mean(poses)))



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
		middle = (((l_hip[0]+r_hip[0])/2), ((l_hip[1]+r_hip[1])/2))

		pairs = get_key_pairs(neck,middle,l_knee,r_knee)
		angle = calculate_angle(pairs, (neck,middle))
		return angle




def get_key_pairs(neck,middle,l_knee,r_knee):
	return (([middle,l_knee]),([middle, r_knee]))




def calculate_angle(pairs, neck_middle):
	pose = []
	for side in pairs:
		angle = 0

		o = math.sqrt((side[0][1] - side[1][1])**2)
		a = math.sqrt((side[0][0] - side[1][0])**2)

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


	angle = 0

	o = math.sqrt((neck_middle[0][1] - neck_middle[1][1])**2)
	a = math.sqrt((neck_middle[0][0] - neck_middle[1][0])**2)

	if (o == 0) & (a == 0) :
		raise ValueError("Keypoints in same place")
	elif a == 0:
		angle += 90
	elif o == 0:
		angle = 0
	else:
		angle += math.degrees(math.atan(o/a))

	pose += angle

	return pose  # The greater the angle the more upright the person




parser = argparse.ArgumentParser()
parser.add_argument('--json', dest='json', help='The json file to visualise')
cfg = parser.parse_args()


with open(cfg.json) as file:
	json_data = json.load(file)




files = preprocessor.get_file_names(json_data)
actual_poses = preprocessor.calculate_poses(json_data, files, 0.3)
actual_poses = np.array(actual_poses)[:,1]

four_key_poses = calculate_poses(json_data, files, 0.3)
four_key_poses = np.array(four_key_poses)[:,1]



error = []
for i in range(len(files)):
	diff = math.sqrt((float(actual_poses[i]) - float(four_key_poses[i]))**2)

	percentage = (diff/float(four_key_poses[i]))*100

	error.append(percentage)

print("mean", statistics.mean(error))
print("max", max(error))
print("min", min(error))
