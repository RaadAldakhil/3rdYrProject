"""

A set of utility functions for calculating various
features of EEG data and plotting graphs

"""

import numpy as np
import time
import pylsl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
from scipy import signal

fig = plt.figure(1)


def connect():
    """
	Connect to an NIC stream is one is available
	and return the stream inlet if successful.
	"""
    stream_name = 'NIC'
    streams = pylsl.resolve_stream('type', 'EEG')

    try:
        for i in range(len(streams)):
            if streams[i].name() == stream_name:
                index = i

        print("NIC stream available")

        print("Connecting to NIC stream... \n")
        return pylsl.StreamInlet(streams[index])

    except NameError:
        print("Error: NIC stream not available\n\n\n")


def recieve_data(inlet, chs, fs, duration):
    """
	Recieve EEG data for the given duration
	"""
    stop_time = time.time() + duration
    data = None

    n_samples = int(round(duration * fs))

    while time.time() < stop_time:
        try:
            sample, timestamp = inlet.pull_sample()
            sample = np.take(sample, chs)
            sample = sample

            if data is None:
                data = np.array([sample])
            else:
                data = np.append(data, [sample], axis=0)

        except KeyboardInterrupt:
            print('Program halted')

    return data


def find_events(data, fs, overlap):
    """
	Find the events in the given data sample
	"""
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    count = int(np.floor((n_samples - fs) / float(overlap)) + 1)
    marks = np.asarray(range(0, count + 1)) * overlap
    marks = marks.astype(int)
    events = np.zeros((500, n_channels, count))

    for e in range(0, count):
        events[:, :, e] = data[marks[e]: marks[e] + fs, :]

    return events


def band_power(PSD, f, low, high):
    """
	Calculate the mean power of given frequency range
	"""
    indices, = np.where((f >= low) & (f < high))
    mean = np.mean(PSD[indices, :], axis=0)
    return mean


def features(data, fs):
    """
	Calculate the frequency features in the provided data
	"""

    sample_length, channels = data.shape

    # Apply Hamming window
    window = np.hamming(sample_length)
    window_without_dc = data - np.mean(data, axis=0)  # Remove offset
    window_centered = (window_without_dc.T * window).T

    NFFT = 1
    while NFFT < sample_length:
        NFFT *= 2

    Y = np.fft.fft(window_centered, n=NFFT, axis=0) / sample_length
    PSD = 2 * np.abs(Y[0:NFFT / 2, :])
    f = fs / 2 * np.linspace(0, 1, NFFT / 2)

    # These are the features we classify on. They are the following frequency
    # bands and variations of these bands: Delta(0-4), Theta(4-8) Alpha (7-12),
    # Beta (12-30).
    delta = band_power(PSD, f, 0, 4)
    theta = band_power(PSD, f, 4, 7)
    alpha = band_power(PSD, f, 7, 12)
    alpha_low = band_power(PSD, f, 7, 9.5)
    alpha_high = band_power(PSD, f, 9.5, 12)
    beta = band_power(PSD, f, 12, 30)

    features = np.concatenate((alpha, alpha_low, alpha_high, beta), axis=0)
    features = np.log10(features)

    return features


def event_features(events, fs):
    """
	Generate a mapping for events against features
	"""
    global mapping
    no_events = events.shape[2]

    for e in range(no_events):
        if e == 0:
            feat = features(events[:, :, e], fs).T
            mapping = np.zeros((no_events, feat.shape[0]))

        mapping[e, :] = features(events[:, :, e], fs).T

    return mapping


def plot_fft(PSDHz, spec_freqs):
    """
	Plot the spectrum frequency
	"""
    print("Generating fft plot")
    spectrum_PSDperHz = np.mean(PSDHz, 1)
    plt.subplot(224)
    plt.plot(spec_freqs, 10 * np.log10(spectrum_PSDperHz))  # dB re: 1 uV
    plt.xlim((0, 60))
    plt.ylim((-30, 50))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD per Hz (dB re: 1uV^2/Hz)')
    plt.grid(True)


def plot_band_power(data, psd, times, freqs, start_freq, end_freq):
    """
	Plot the band power
	"""
    print("Plotting band power over time. Frequency range: " + str(start_freq) + " - " + str(end_freq))
    indices = (freqs > start_freq) & (freqs < end_freq)
    band_power = np.sqrt(np.amax(psd[indices, :], 0))
    plt.subplot(223)
    plt.plot(times, band_power)
    plt.ylim([np.amin(band_power), np.amax(band_power) + 1])
    plt.xlabel('Time (sec)')
    plt.ylabel('EEG Amplitude (uVrms)')
    plt.grid(True)


def plot_spectrogram(data, psd, times, freqs):
    """
	Generate the heatmap / spectrogram
	"""
    print("Generating spectrogram...")
    f_lim_Hz = [0, 30]  # Frequency limits for plotting
    plt.subplot(222)
    plt.pcolor(times, freqs, 10 * np.log10(psd))
    plt.clim([-25, 26])
    plt.xlim(times[0], times[-1] + 1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True)


def bandpass(data, fs, start, stop):
    """
	Bandpass filtering
	"""
    bp_Hz = np.zeros(0)
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(3, bp_Hz / (fs / 2.0), 'bandpass')
    print('Bandpass filtering to: ' + str(bp_Hz[0]) + "-" + str(bp_Hz[1]) + " Hz")
    return signal.lfilter(b, a, data, 0)


def get_spectrum_data(data, fs, NFFT, overlap):
    """
	Calculate the spectrum data (PSD per Hz)
	"""
    print("Calculating spectrum data")
    psd_per_hz, freqs, times = mlab.specgram(np.squeeze(data), NFFT=NFFT, window=mlab.window_hanning, Fs=fs,
                                             noverlap=overlap)
    psd = psd_per_hz * fs / float(NFFT)
    return psd_per_hz, psd, freqs, times


def signalplot(data, times, r):
    """
	Plot the signal
	"""
    print("Generating signal plot")
    plt.subplot(221)
    plt.plot(times[r:], data[r:])
    plt.xlabel('Time (sec)')
    plt.ylabel('Power (uV)')
    plt.title('Signal')


def remove_dc(data, fs):
    """
	Remove the DC offset
	"""
    cutoff = 1.0
    print("Highpass filtering at: " + str(cutoff) + " Hz")
    b, a = signal.butter(2, cutoff / (fs / 2.0), 'highpass')
    data = signal.lfilter(b, a, data, 0)
    return data


def notch_filter(data, fs):
    """
	Remove mains interference
	"""
    notch_freq = np.array([48.0])
    for f in np.nditer(notch_freq):
        stop = f + 3.0 * np.array([-1, 1])
        b, a = signal.butter(3, stop / (fs / 2.0), 'bandstop')
        data = signal.lfilter(b, a, data, 0)

    return data


def reject_outliers(data, m=2):
    """
	Remove data outliers
	"""
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def plot_data(f, ch, fs):
    """
	Generates the plots for the given csv file
	"""
    data = pd.read_csv(f)
    data = data[ch]
    data = notch_filter(data, fs)
    data = bandpass(data, fs, 1.0, 50.0)
    times = np.arange(len(data)) / float(fs)
    signalplot(data, times, 1000)

    NFFT = 1
    while NFFT < len(data):
        NFFT *= 2

    overlap = NFFT - int(0.25 * fs)
    psd_per_hz, psd, freqs, times = get_spectrum_data(data, fs, NFFT, overlap)

    plot_spectrogram(data, psd, times, freqs)
    plot_band_power(data, psd, times, freqs, 8, 12)
    plot_fft(psd_per_hz, freqs)
    fig.savefig('plots.png')
