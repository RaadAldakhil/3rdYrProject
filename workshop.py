"""

This is an example script for a workshop activity. In this workshop one would use the 
command line to make the activity interactive and more educational. This script
demonstrates the functionality that can be presented in the workshop, including 
gather training data, gathering testing data, calculating the precision/accuracy 
of the classifier and then running the classifier against live data to play music
when the predictions are correct.

"""

from eeg import EEG

if __name__ == '__main__':
    # Create the classifier/eeg class
    clf = EEG()

    # Record some training data
    clf.record_eeg('Eyes Open', 5)
    clf.record_eeg('Eyes Closed', 5)
    clf.record_eeg('Eyes Open', 5)
    clf.record_eeg('Eyes Closed', 5)

    # Record some testing data
    clf.record_test('Eyes Open', 5)
    clf.record_test('Eyes Closed', 5)

    # Create/Fit our classifier
    clf.create_classifier('Eyes Open', 'Eyes Closed')

    # Test our classifiers accuracy
    print('Accuracy for Eyes Open: %f' % clf.test('Eyes Open', 'Eyes Open'))
    print('Accuracy for Eyes Closed: %f' % clf.test('Eyes Closed', 'Eyes Closed'))

    # Run live data against music
    clf.run('Eyes Closed')

"""
	# EEGLearner
	# EEGLearner.addClassifier()
	# EEGLearner.addFeature() -> Eye blink

	sphero_controller = EEGLearner()

	
	###### Play MUSIC!

	classifiers = EEGClassifier('Alpha classifier')
	classifiers.get_training_data('Eyes Open', 10)
	classifiers.get_training_data('Eyes Closed', 10)
	classifiers.get_training_data('Eyes Blink', 0.5)

	classifiers.create_classifier('Eyes Closed', 'Eyes Open', 'Eyes Closed')
	classifiers.create_classifier('Eyes Blink', 'Eyes Open', 'Eyes Closed')

	# Read Data


	# Step 1: Create our EEG Learner Class

	# Step 2: Get some training data

	# Step 3: Create our classifiers

	# Step 4: Run live data against our classifiers
	try:
		data = object.read_data(0.5)"""

"""
import time
import numpy as np # Makes matrix/array calculations easy
import eeg_utils as eeg

if __name__ == '__main__':
	eeg.play_beep()
	inlet = eeg.connect()
	eeg.play_beep()
	train1 = eeg.recieve_data(inlet, 10)
	eeg.play_beep()
	train2 = eeg.recieve_data(inlet, 10)
	eeg.play_beep()

	print(train1.shape)

	epoch1 = eeg.epochs(train1, 500.0, 250.0)
	epoch2 = eeg.epochs(train2, 500.0, 250.0)

	feats1 = eeg.epoch_bands(epoch1, 500.0)
	feats2 = eeg.epoch_bands(epoch2, 500.0)

	[classifier, mu_ft, std_ft] = eeg.train_classifier(feats1, feats2)

	data_buffer = np.zeros((500.0 * 5, 10))
	decision_buffer = np.zeros((5, 1))

	previous = None
	previous2 = None
	playing = False

	try:
		while True:
			test_data = eeg.recieve_data(inlet, 1)

			feat_vector = eeg.compute_feature_vector(test_data, 500.0)
			y_hat = eeg.test_classifier(classifier, feat_vector, mu_ft, std_ft)
			print(y_hat[0])
			print(y_hat)

			if int(y_hat[0]) == 1 and previous == 1 and playing == False:
				playing = True
				eeg.play_music()
			elif int(y_hat[0]) == 0 and previous == 0 and playing == True:
				playing = False
				eeg.stop_music()

			previous = y_hat[0]

			print str(y_hat)

	except KeyboardInterrupt:
		print('moew')"""
