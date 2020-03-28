from classification import classifier
from utils import config
import pickle
import os

#TRAINING CONFIG
final_classifier_score = 0
final_classifier = 0


parser = argparse.ArgumentParser()

parser.add_argument('--config', dest='config', help='The config file to initialise the training with')

cfg = parser.parse_args()


print("Loading configuration...", end="")
args = config.get_parser(cfg.config)
print("Done")


#TRAINING
print("Begining training process with", args.ITERATIONS, "interations")

for c in range(args.ITERATIONS):
	nn_classifier = classifier.Neural_Network(args.TRAIN_DATASET, args.THRESHOLD)
	score = nn_classifier.score(args.TRAIN_DATASET)

	if score > final_classifier_score: # Should be changed to test dataset
		final_classifier_score = score
		final_classifier = nn_classifier


print("Trained with score of...", final_classifier_score)


#SAVING MODEL
FILENAME = str(args.ITERATIONS) + "_epoch_trained.sav"

print("Saving final model")


pickle.dump(final_classifier, open(os.path.join(args.MODEL_OUTPUT_DIR,FILENAME), 'wb'))


print("Saved final model to", os.path.join(args.MODEL_OUTPUT_DIR,FILENAME))
