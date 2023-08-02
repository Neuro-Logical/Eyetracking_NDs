__author__ = "Miguel A Iglesias"
__date__ = "June Fourteenth, 2022"
__version__ = "v0.1"
__description__ = \
"""
DESCRIPTION

    version 0.1

    This code determines when someone begins to speak via three methods. By
    comparing the hilbert envelope to a VAD to alignment data from a Grad Student,
    a good start time is found.

    The code also determines whether or not someone says what they should.

    The output is a file with the file name, what word they said, and the start time.
"""

import os
import contextlib
import wave
import numpy as np
import csv
import pvcobra
import struct
from scipy.signal import butter, filtfilt, hilbert
from statistics import mean, stdev
from argparse import *
import sys

#Creates an instance of the Cobra VAD
cobra = pvcobra.create(access_key='xbSp/A7vbP5gvclgJwnoFsZytXgemy1qYvS5ETBjZKNeqJe1QuraeA==')

def wav_info(path, file, align_names, align_times):
    '''
    Takes a file and has it read by the read_file function. For each 30 ms frame
    the sound probability is determined. Possible start times (PSTs) are determined by
    feeding the probabilities into the z_score_indexing fucntion. These PSTs are
    thwn used to find a more precise start time from the hilbert envelope
    These are then compared to alignment data by best_start_time to return one
    start_time.
    '''

    is_sound = []
    audio = read_file(path + file, cobra.sample_rate)
    num_frames = len(audio) // cobra.frame_length

    for i in range(num_frames):
        frame = audio[i * cobra.frame_length:(i + 1) * cobra.frame_length]
        result = cobra.process(frame)
        is_sound.append(result)

    possible_start_times = z_score_indexing(is_sound, file)

    amp_env = amp_envelope(audio)

    env_times = env_z_scores(amp_env, possible_start_times)

    start_time, method = best_start_time(possible_start_times, env_times, path, file, align_names, align_times)

    return start_time, file[:-4], method

def read_file(file_name, sample_rate):
    '''
    Reads in a file and returns it in 30 ms frames.
    '''
    wav_file = wave.open(file_name, mode="rb")
    channels = wav_file.getnchannels()
    num_frames = wav_file.getnframes()

    if wav_file.getframerate() != sample_rate:
        raise ValueError("Audio file should have a sample rate of %d. got %d" % (sample_rate, wav_file.getframerate()))

    samples = wav_file.readframes(num_frames)
    wav_file.close()

    frames = struct.unpack('h' * num_frames * channels, samples)

    if channels == 2:
        print("Picovoice processes single-channel audio but stereo file is provided. Processing left channel only.")

    return frames[::channels]

def amp_envelope(signal):
    '''
    Creates a pre-filtered hilbert envelope
    '''
    analyticSignal = hilbert(signal)
    amplitudeEvelope = np.abs(analyticSignal)
    return FilteredSignal(amplitudeEvelope, 16000, 20)

def FilteredSignal(signal, fs, cutoff):
    '''
    Filters the hilbert envelope
    '''
    B, A = butter(1, cutoff / (fs / 2), btype='low')
    filtered_signal = filtfilt(B, A, signal, axis=0)
    return filtered_signal

def z_score_indexing(sound_probabilities, file_name):
    '''
    To determine when the sound starts, we look at standard deviations based on
    silence which is generally at the beginning. Sometimes, silence is checked
    at the end. Returns the indices of areas with high z-scores compared to
    silence. If no silence at end or beginning, an empty list is returned
    '''

    n = 12

    segments = [sound_probabilities[i:i + n] for i in range(0, len(sound_probabilities), n)]

    means = []


    for segment in segments:
        means.append(mean(segment))

    min_index = means.index(min(means))

    z_score_index = []

    z_score_start = z_scores(sound_probabilities, segments[min_index])

    if max(z_score_start) > 20:
        for i in range(len(z_score_start)):
            if z_score_start[i] > 20:
                z_score_index.append(i)

    return cleaner(z_score_index)

def z_scores(array, param):
    '''
    Calculates the z score for each point based on silence.
    '''
    z_score = [None] * len(array)
    avg = mean(param)
    sd = stdev(param)
    for i in range(len(array)):
        z_score[i] = (array[i] - avg)/sd
    return z_score

def cleaner(unclean_list):
    '''
    Sometimes an index is duplicated so this deletes the unwanted ones.
    '''
    unwanted_indices = []
    for i in range(len(unclean_list)):
        if i != len(unclean_list) - 1:
            if unclean_list[i] == unclean_list[i+1]:
                unwanted_indices.append(i)
        if unclean_list[i] < unclean_list[i-1] and i != 0:
            unwanted_indices.append(i)
    for index in sorted(unwanted_indices, reverse=True):
        del unclean_list[index]
    clean_list = unclean_list
    return starting_times(clean_list)

def starting_times(indices):
    '''
    Indices of high scores are given. We only want to the first per peak, so this
    returns the first one in each peak assuming in a peak the indices are within
    10 of each other. Returns what times peaks with high z-scores start
    '''
    holder = 0
    start_indices = []

    if len(indices) == 1:
        start_indices.append(indices[0])

    else:
        for i in range(len(indices)):
            if i != len(indices) - 1:
                if indices[i+1] - indices[i] > 10:
                    start_indices.append(indices[holder])
                    holder = i + 1
            else:
                start_indices.append(indices[holder])
    return [(start_index + 1) * (512/16000) for start_index in start_indices]

def env_z_scores(env, pst):
    '''
    Determines when there is the most silence and gets z-scores for the whole
    envelope.
    '''
    n = 4000

    segments = [env[i:i + n] for i in range(0, len(env), n)]

    means = []

    for segment in segments:
        means.append(mean(segment))

    min_index = means.index(min(means))

    silence = segments[min_index]

    z_score = z_scores(env, silence)

    return env_start_times(z_score, pst)

def env_start_times(z_scores, pst):
    '''
    Determines start time by seeing when the z-score eclipses 50
    '''
    env_pst = []
    for val in pst:
        peak = z_scores[(int(val*16000) - 2560):(int(val*16000) + 2560)]
        for i in range(len(peak)):
            if i == len(peak) - 1:
                pass
            elif peak[i] < 50 and peak[i+1] > 50:
                env_pst.append(z_scores.index(peak[i+1]))
    return env_pst


def best_start_time(vad_start_times, env_times, path, wavf, align_names, align_times):
    '''
    Compares start time options and returns the option closet to the alignment
    data in lines. Requires the wavfile name to compare to the txt file. The method
    is also returned. {1: No good start times, 2: Determined with envelope and
    alignment, 3: envelope no alignment, 4: 2 but VAD, 5: 3 but VAD}
    '''

    if len(env_times) == 0 and len(vad_start_times) == 0:
        return 'na', 1

    else:
        distances = []
        if len(env_times) != 0:
            if wavf[3:-4] in align_names:
                indx = align_names.index(wavf[3:-4])
                for i in env_times:
                    if align_times[indx] != 'None':
                        distances.append(abs(i/16000 - float(align_times[indx])))
                    else:
                        return env_times[0]/16000, 3
            else:
                return env_times[0]/16000, 3

        if len(distances) != 0:
            return env_times[distances.index(min(distances))]/16000, 2

        if len(vad_start_times) != 0 and len(distances) == 0:
            if wavf[3:-4] in align_names:
                indx = align_names.index(wavf[3:-4])
                for i in vad_start_times:
                    if align_times[indx] != 'None':
                        distances.append(abs(i - float(align_times[indx])))
                    else:
                        return vad_start_times[0], 5
            else:
                return vad_start_times[0], 5
        if len(distances) != 0:
            return vad_start_times[distances.index(min(distances))], 4

def data_reorganize(data, path):
    '''
    To make the CSV, the data is organized into a list with the data for every
    observation.
    '''
    reorganized_data = [None] * len(data[0])

    for i in range(len(data[0])):
        reorganized_data[i] = [data[j][i] for j in range(len(data))]

    return csv_maker(reorganized_data, path)

def csv_maker(all_values, path):
    '''
    Outputs the data to a CSV file
    '''
    with open(path + "start_times.csv", "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(all_values)
    return

def start(args):

    '''
    Sends the data for analysis and then sends it to be created into a table
    Data is a list of lists containing the file name, what word they said, and
    their suspected start time.
    '''

    data = [[],[],[]]

    alignment = open(args[1])
    lines = alignment.readlines()

    align_names = []
    align_times = []

    for line in lines:
        split = line[:-1].split(' ')
        if split[0][-8:] == '-16k.wav':
            align_names.append(split[0][:-8])
        else:
            align_names.append(split[0][:-4])
        align_times.append(split[1])

    for file in sorted(os.listdir(args[0])):
        if file.endswith('.wav') == False:
            pass
        else:
            print(file[:-4])
            start_time, file_name, start_time_method = wav_info(args[0], file, align_names, align_times)
            data[0].append(file_name)
            data[1].append(start_time)
            data[2].append(start_time_method)

    data_reorganize(data, args[0])

    return

def main():
    """
    function to call if run from command line
    """
    # Define arguments for parser
    parser = ArgumentParser(description = __description__,
                        formatter_class = RawDescriptionHelpFormatter)

    parser.add_argument('dir1', action="store", default=None,
                        help='Input directory of given sound files')
    parser.add_argument('dir2', action="store", default=None,
                        help='Input file with given alignment times')

    args = parser.parse_args()

    start([args.dir1, args.dir2])

# if run from command line
if __name__ == "__main__":
    main()


#This function resampled each file from 24000 to 16000 and saved them
#into a new folder
'''
import librosa
import soundfile as sf
def converter(file):
    print(file)
    data, sr_new = librosa.load(dir1 + file, sr = 16000)
    sf.write(dir2 + '16_' + file, data, sr_new)

for file in os.listdir(dir1):
    if file.endswith('.wav'):
        converter(file)
'''
