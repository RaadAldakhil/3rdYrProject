"""

EEG Classifier class to provide basic functionality
for predicting EEG state dependent on frequency features.

"""

import numpy as np
import time
import winsound
import utils
from sklearn import svm


class EEG():

    def __init__(self):
        """
		Initialise our class variables and set up a connection
		to the Enobio EEG
		"""
        self.recording_chs = np.arange(8)
        self.fs = 500
        self.overlap = 250
        self.labels = dict()
        self.feats = {'train': dict(), 'test': dict()}
        self.connect()
        self.music = 0
        self.musicA = "C:\Users\Outreach\Desktop\Reworked_workshop\FDP.wav"
        self.musicB = "C:\Users\Outreach\Desktop\Reworked_workshop\TD.wav"
        self.musicC = "C:\Users\Outreach\Desktop\Reworked_workshop\V.wav"

    def connect(self):
        """
		Connect to our Enobio 8 stream
		"""
        self.inlet = utils.connect()

    def set_channels(self, channels):
        """
		Set the channels that are recording
		"""
        self.channels = channels

    def get_channels(self):
        """
		Return the set channels
		"""
        return self.channels

    # def set_recording_channels(self, channels):
    #    """
    #	Set the channels to listen to
    #	"""
    #   indices = [i for i, x in enumerate(self.channels) if x < a or x > b]
    #  self.recording_chs = indices

    def get_recording_channels(self):
        """
		Return the channel indices we are listening to
		"""
        return self.recording_chs

    def set_music(self, music_file):
        """
		Set the music for the runtime activity
		"""
        self.music = music_file

    def get_music(self):
        """
		Return our set music file
		"""
        return self.music

    def toggle_music(self):
        """
		Toggle the music on/off
		"""
        f = None
        winsound.Beep(300, 300)
        if int(self.music) <= int(1):
            f = self.musicA
        elif int(self.music) <= int(3):
            f = self.musicB
        elif int(self.music) <= int(5):
            f = self.musicC
        winsound.PlaySound(f, winsound.SND_ASYNC)
        self.music = 0

    def play_beep(self, f=500, d=500):
        """
		Play a beep sound to indicate start/end of recording
		"""
        winsound.Beep(500, 500)

    def record_eeg(self, label, duration):
        """
		Record EEG activity for the given label and
		specified duration. Store in either the testing
		or training dict dependent on train_or_test
		"""
        print('----------------------\n')
        print(' Recording training for %s' % label)
        print(' Duration: %f seconds\n' % duration)

        time.sleep(2)
        self.play_beep()
        data = utils.recieve_data(self.inlet, self.recording_chs, self.fs, duration)
        events = utils.find_events(data, self.fs, self.overlap)
        feats = utils.event_features(events, self.fs)

        if label in self.feats['train']:
            feats = np.concatenate((self.feats['train'][label], feats), axis=0)

        self.feats['train'][label] = feats

        print(' Recording finished\n')
        print('----------------------\n')
        self.play_beep()
        time.sleep(2)

    def record_test(self, label, samples):
        """
		Record N samples of training data for the 
		given label
		"""
        print('----------------------\n')
        print(' Recording testing for %s' % label)
        print(' Duration: %f seconds\n' % samples)

        if label in self.feats['test']:
            feats = self.feats['test'][label]
        else:
            feats = []

        time.sleep(2)
        self.play_beep()
        for i in range(samples):
            feats.append(utils.recieve_data(self.inlet, self.recording_chs, self.fs, 1))

        self.feats['test'][label] = feats

        print(' Recording finished\n')
        print('----------------------\n')
        self.play_beep()
        time.sleep(2)

    def test(self, sample_label, actual_label):
        """
		Test the accuracy of our classifier
		"""
        c = self.labels[actual_label]
        X = self.feats['test'][sample_label]
        accuracy = 0

        for x in X:
            data = utils.features(x, self.fs)
            prediction = self.make_prediction(data)
            if prediction == c:
                accuracy = accuracy + 1

        accuracy = accuracy / float(len(X))
        return accuracy

    def scale(self, data):
        """
		Feature scaling on the data by subtracting
		the stored mean and dividing by the stored std
		"""
        X = (data - self.scale_mean) / self.scale_std
        return X

    def create_classifier(self, label1, label2):
        """
		Create a binary classifier for training sets
		"""
        self.labels[label1] = 0
        self.labels[label2] = 1

        class_one = np.zeros((self.feats['train'][label1].shape[0], 1))
        class_two = np.ones((self.feats['train'][label2].shape[0], 1))

        Y = np.concatenate((class_one, class_two), axis=0)
        Y = np.ravel(Y)

        data = np.concatenate((self.feats['train'][label1], self.feats['train'][label2]), axis=0)
        self.scale_mean = np.mean(data)
        self.scale_std = np.std(data)
        X = self.scale(data)

        self.clf = svm.SVC()
        self.clf.fit(X, Y)

    def make_prediction(self, data):
        """
		Use the classifier to make a prediction
		"""
        X = self.scale(data)
        X = X.reshape(1, -1)
        return int(self.clf.predict(X))

    def run(self, label=0):
        """
		Collects live EEG data and plays music dependent on the
		predicted label
		"""

        print('----------------------\n')
        print(' Running classifier against real time data')
        print(' Press Ctrl+C to halt\n')

        counter = 0
        action_label = label
        if label != 0:
            action_label = self.labels[label]

        try:
            while True:
                data = utils.recieve_data(self.inlet, self.recording_chs, self.fs, 1)
                data = utils.features(data, self.fs)
                prediction = self.make_prediction(data)
                print(' Predicting: %d' % prediction)
                winsound.Beep(600, 100)
                if prediction:
                    self.music += 1
                    counter += 1
                else:
                    counter += 1
                if counter >= 7:
                    counter = 0
                    self.toggle_music()

                prev_prediction = prediction

        except KeyboardInterrupt:
            print(' Recording halted\n')
            print('----------------------\n')
