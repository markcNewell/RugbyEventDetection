from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from pose_estimation import preprocessor
import math, json




class Neural_Network:
	"""docstring for ClassName"""
	def __init__(self, filename, hidden_layers=(6,)):
		self.clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layers, max_iter=1000000)
		self.le = preprocessing.LabelEncoder()

		x,y = self.split_dataset(filename)
		y = self.le.fit_transform(y)
		self.train(x,y)

		print(self.clf.score(x,y), end="...")


	def train(self, dataset, results):
		self.clf.fit(dataset, results)


	def predict(self, json_data):
		files = preprocessor.get_file_names(training_data)

		if files == 1:
			pose = preprocessor.calculate_poses(training_data, files)[0]
			ratio = preprocessor.get_attr(training_data, files, 'ratio')[0]
			return self.clf.predict([pose,ratio])
		else:
			raise ValueError("Too many values in alphapose json")


	def split_dataset(self, filename):
		with open(filename) as file:
			training_data = json.load(file)


		files = preprocessor.get_file_names(training_data)
		poses = preprocessor.calculate_poses(training_data, files)
		ratios = preprocessor.get_attr(training_data, files, 'ratio')
		tags = preprocessor.get_attr(training_data, files, 'tag')

		return [list(a) for a in zip(poses,ratios)], tags



		