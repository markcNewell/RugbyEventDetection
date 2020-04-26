import json, argparse
import numpy as np
import preprocessor
import matplotlib.pyplot as plt
from textwrap import wrap



parser = argparse.ArgumentParser()
parser.add_argument('--json', dest='json', help='The json file to visualise')
cfg = parser.parse_args()



with open(cfg.json) as file:
	json_data = json.load(file)



files = preprocessor.get_file_names(json_data)
poses = preprocessor.calculate_poses(json_data, files, 0.3)
poses = np.array(poses)[:,1]
ratios = preprocessor.get_attr(json_data, files, 'ratio')
tags = preprocessor.get_attr(json_data, files, 'tag')

print(len(poses))
print(len(files))

rucks, mauls, lineouts, scrums = ([] for i in range(4))

for i in range(len(files)):
	if (tags[i] == "ruck"):
		rucks.append([float(poses[i]),float(ratios[i])])
	elif (tags[i] == "maul"):
		mauls.append([float(poses[i]), float(ratios[i])])
	elif (tags[i] == "scrum"):
		scrums.append([float(poses[i]), float(ratios[i])])
	elif (tags[i] == "lineout"):
		lineouts.append([float(poses[i]), float(ratios[i])])




rucks = np.array(rucks)
mauls = np.array(mauls)
lineouts = np.array(lineouts)
scrums = np.array(scrums)

data = [rucks,mauls,lineouts,scrums]




fig, ax = plt.subplots()

colors = ['b', 'r', 'g', 'm']
tags = ['ruck', 'maul', 'lineout', 'scrum']

for i in range(len(colors)):
	x = data[i][:,1]
	y = data[i][:,0]
	ax.scatter(x, y, c=colors[i], label=tags[i])

ax.legend()
ax.grid(True)

ax.set_ylabel('Mean average pose angle of cluster (degrees)')
ax.set_xlabel('Cluster width:height ratio')
ax.set_title("\n".join(wrap('A graph depicting the average pose against the width:height ratio for each cluster')))

plt.savefig("test.png")
