from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from pose_estimation import preprocessor
import math, json, pickle




class Neural_Network:
	"""docstring for ClassName"""
	def __init__(self, filename, threshold, hidden_layers=(6,)):
		self.le = preprocessing.LabelEncoder()
		self.threshold = threshold

		self.clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layers, max_iter=1000000)
		x,y = self.split_dataset(filename)
		y = self.le.fit_transform(y)
		self.train(x,y)


	def train(self, dataset, results):
		self.clf.fit(dataset, results)


	def score(self, filename):
		x,y = self.split_dataset(filename)
		y = self.le.fit_transform(y)
		return self.clf.score(x,y)


	def predict(self, json_data):
		files = preprocessor.get_file_names(json_data)

		if len(files) == 1:
			poses = preprocessor.calculate_poses(json_data, files, self.threshold)
			if len(poses) > 0:
				ratio = preprocessor.get_attr(json_data, files, 'ratio')[0]
				return self.clf.predict_proba([[poses[0],ratio]])
			else:
				raise ValueError("Poses not found for file")
		else:
			raise ValueError("Too many values in alphapose json")


	def split_dataset(self, filename):
		with open(filename) as file:
			training_data = json.load(file)


		files = preprocessor.get_file_names(training_data)
		poses = preprocessor.calculate_poses(training_data, files, self.threshold)
		ratios = preprocessor.get_attr(training_data, files, 'ratio')
		tags = preprocessor.get_attr(training_data, files, 'tag')

		return [list(a) for a in zip(poses,ratios)], tags



		