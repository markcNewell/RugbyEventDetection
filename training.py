from classification import classifier
import pickle
import os

#TRAINING CONFIG
ITERATIONS = 10
TRAIN_DATASET = "./dataset/train/annotations/train.json"
MODEL_OUTPUT_DIR = "./trained/classification/"
final_classifier_score = 0
final_classifier = 0


#TRAINING
print("Begining training process with", ITERATIONS, "interations")

for c in range(ITERATIONS):
	nn_classifier = classifier.Neural_Network(TRAIN_DATASET, training=True)
	score = nn_classifier.score(TRAIN_DATASET)

	if score > final_classifier_score: # Should be changed to test dataset
		final_classifier_score = score
		final_classifier = nn_classifier


print("Trained with score of...", final_classifier_score)


#SAVING MODEL
FILENAME = str(ITERATIONS) + "_epoch_trained.sav"

print("Saving final model")


pickle.dump(final_classifier, open(os.path.join(MODEL_OUTPUT_DIR,FILENAME), 'wb'))


print("Saved final model to", os.path.join(MODEL_OUTPUT_DIR,FILENAME))
