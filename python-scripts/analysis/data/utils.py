import glob

import librosa
import pandas as pd
import argparse
import sys, os

from scipy.io import wavfile

from audio_utils import removeNoise, split_audio_chunks, filter_voice, ToolGammatoneFb, gcc_gammatoneFilter, \
    butter_lowpass_filter, pitch_augment
import scipy.io.wavfile
import numpy as np
from tqdm.auto import tqdm, trange
import multiprocessing

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from CONFIG import *
import pickle
import random




def create_chunk_audio_background(audio_filename, output_dir, length_audio):
    """
    Create from a wav file chunk of length audio
    :param audio_filename: input audio filename
    :param output_dir: output directory for saving
    :param length_audio: length in milliseconds of the chunks
    :return: the new dataset with all the chunks audio filename
    """

    df = pd.DataFrame(columns=['audio_filename', 'labels'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample_rate, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename,
                                                                       size_chunks=length_audio,
                                                                       overlap=CHUNCK_OVERLAP)

    for i, (audio_c1, audio_c2) in enumerate(zip(chunks_channel1, chunks_channel2)):
        audio_filename = f"background_sample_reddy_{i}.wav"
        filename_write = os.path.join(output_dir, audio_filename)

        data = np.stack((audio_c1, audio_c2), axis=1)

        df = df.append({"audio_filename": audio_filename, "labels": -1}, ignore_index=True)

        if not os.path.exists(filename_write):
            scipy.io.wavfile.write(filename_write, sample_rate, data)


    df.to_csv(f"background-{length_audio}.csv", index=False)

    return df


def create_chunk_audio(df, output_dir, length_audio):
    """
    Create from a wav file chunk of length audio if they contain x% of voice and saved them
    :param df: input dataset of wav filename
    :param output_dir: output directory for saving
    :param length_audio: length in milliseconds of the chunks
    :return: the new dataset with all the chunks audio filename
    """

    new_df = pd.DataFrame()
    i = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, item in tqdm(df.iterrows()):
        audio_filename = item['audio_filename']
        end_audio = item['stop_audio_timestamp'] - item['start_audio_timestamp']
        sample_rate, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename, end_audio,
                                                                           size_chunks=length_audio,
                                                                           overlap=CHUNCK_OVERLAP)

        for j, (signal1, signal2) in enumerate(zip(chunks_channel1, chunks_channel2)):
            filename = str(index) + '-' + str(j) + '_' + str(item['subject_id']) + '.wav'
            filename_write = os.path.join(output_dir, filename)

            data = np.stack((signal1, signal2), axis=1)

            if not filter_voice(signal1, sample_rate):
                print(f"{filename} not containing voice")
                # scipy.io.wavfile.write("/tmp/no_voice/" + filename, sample_rate, data)

            else:
                new_df = new_df.append(item, ignore_index=True)
                new_df.at[i, 'audio_filename'] = filename
                if not os.path.exists(filename_write):
                    scipy.io.wavfile.write(filename_write, sample_rate, data)
                i += 1

    new_df.to_csv("chunck_dataset-{}.csv".format(length_audio), index=False)

    return df


def create_mel_spectogram(filename):
    if not os.path.exists(filename.split('.wav')[0]):
        filename = os.path.join(PATH_DATA, filename)
        fs, signal = scipy.io.wavfile.read(filename, "wb", )

        signal1 = signal[:, 0]

        mel_spec = librosa.feature.melspectrogram(signal1, fs)

        pickle_filename = filename.split('.wav')[0]
        pickle.dump(mel_spec, open(pickle_filename, "wb"))

    return True


def create_gammatone(filename):
    """
    Create Gammatone filter bank and apply it to a wav file define by filename
    :param filename: wav file
    :return:
    """
    if not os.path.exists(filename.split('.wav')[0]):
        filename = os.path.join(PATH_DATA, filename)
        fs, signal = scipy.io.wavfile.read(filename, "wb", )

        signal1 = signal[:, 0]
        signal2 = signal[:, 1]

        signal1 = butter_lowpass_filter(signal1, 1000, fs)
        signal2 = butter_lowpass_filter(signal2, 1000, fs)

        gamma_sig1 = ToolGammatoneFb(signal1, fs, iNumBands=NUM_BANDS)
        gamma_sig2 = ToolGammatoneFb(signal2, fs, iNumBands=NUM_BANDS)

        gammatone = np.stack((gamma_sig1, gamma_sig2), axis=-1)

        pickle_filename = filename.split('.wav')[0]
        pickle.dump(gammatone, open(pickle_filename, "wb"))

    return True


def remove_noise_data(audio_filename):
    fs, data = wavfile.read(audio_filename)

    rate, data_noise = wavfile.read(PATH_NOISE)
    data_noise = data_noise[:len(data[:, 0]), 0]
    data_noise = data_noise / 32768

    signal1 = data[:, 0] / 32768
    signal2 = data[:, 1] / 32768

    signal1 = removeNoise(signal1, data_noise)
    signal2 = removeNoise(signal1, data_noise)

    filter_noise_data = np.stack((signal1, signal2), axis=-1)
    filename_write = audio_filename.split('.wav')[0] + "_nonoise.wav"
    scipy.io.wavfile.write(filename_write, rate, filter_noise_data)


def create_gammatone_gcc(filename):
    """
    Create Gammatone filter bank with the GCC-PHAT and apply it to a wav file define by filename
    :param filename: wav file
    :return:
    """
    if not os.path.exists(filename.split('.wav')[0]):
        filename = os.path.join(PATH_DATA, filename)
        fs, signal = scipy.io.wavfile.read(filename, "wb", )

        signal1 = signal[:, 0]
        signal2 = signal[:, 1]

        delays, gcc = gcc_gammatoneFilter(signal1, signal2, fs, NUM_BANDS, N_DELAY)

        pickle_filename = filename.split('.wav')[0]
        pickle.dump(gcc, open(pickle_filename, "wb"))

    return True


def process_subject(path_subject, subject_id, center_threshold=20):
    audio_samples_list = glob.glob(os.path.join(path_subject, "2020*/*.wav"))
    angles_filename = glob.glob(os.path.join(path_subject, "logAngles/data.log"))[0]
    head_state_filename = glob.glob(os.path.join(path_subject, "logHeadState/data.log"))[0]

    df_head_angles = pd.read_csv(angles_filename, sep=' ', names=COLUMNS_ANGLES)
    df_head_joints = pd.read_csv(head_state_filename, sep=' ', names=COLUMNS_JOINTS)

    # Drop unnecessary columns
    df_head_angles = df_head_angles.drop(['index', 'vergence'], axis=1)
    df_head_joints = df_head_joints.drop(['index'], axis=1)

    print("Found {} audio samples ".format(len(audio_samples_list)))

    df_data = pd.DataFrame(columns=NEW_COLUMNS_DATA)

    for index, file_name in enumerate(audio_samples_list):

        row_data = {'subject_id': str(subject_id), 'audio_filename': str(file_name),
                    "duration": librosa.get_duration(filename=file_name)}

        if 'nonoise' in file_name:
            os.remove(file_name)
            continue

        # Isolate the data for the length of the audio
        tmp = file_name.split('/')[-1]
        start_audio_timestamp = float(tmp.split('.wav')[0].split('_')[0])
        end_audio_timestamp = float(tmp.split('.wav')[0].split('_')[-1])
        row_data["start_audio_timestamp"] = start_audio_timestamp

        # Get the the final angles
        df_tmp_angles = df_head_angles[df_head_angles['timestamp'] >= int(end_audio_timestamp)]
        index_angle_label_azimuth = abs(df_tmp_angles['timestamp'] - end_audio_timestamp).idxmin()
        azimuth_label = float(df_head_angles.iloc[[index_angle_label_azimuth], 1])

        # Rectified the final azimuth angle with the joint value of the head
        df_tmp_joints = df_head_joints.query(
            'timestamp >= @start_audio_timestamp and timestamp <= @end_audio_timestamp').copy(deep=True)

        if df_tmp_joints.empty:
            print(f"Error with file {file_name} skipping")
            continue

        df_tmp_joints['rectified_azimuth'] = azimuth_label - (df_tmp_joints['joint2'] * -1)

        # Find when the azimuth label change
        initial_value_azimuth = df_tmp_joints.iloc[0, -1]
        df_tmp_joints['diff'] = initial_value_azimuth - df_tmp_joints['rectified_azimuth']
        diff_timestamp = df_tmp_joints.query("rectified_azimuth < @center_threshold")
        if diff_timestamp.empty or df_tmp_joints['diff'].mean() < 1:
            row_data["stop_audio_timestamp"] = end_audio_timestamp
            row_data['azimuth'] = df_tmp_joints['rectified_azimuth'].mean()

        else:
            new_stop_audio_timestamp = diff_timestamp.iloc[0, 0]
            row_data["stop_audio_timestamp"] = new_stop_audio_timestamp
            row_data['azimuth'] = azimuth_label

        row_data["elevation"] = df_tmp_angles.iloc[0, 2]

        df_data = df_data.append(row_data, ignore_index=True)

    df_data.reset_index(drop=True, inplace=True)

    return df_data


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return True


###################################################################################
#                                   MAIN PROCESS                                  #
##################################################################################

def main(args):
    list_subjects = os.listdir(args.data_dir)

    print("Found {} subjects to analyze".format(len(list_subjects)))
    print("----------------------------------------------------------------------\n")

    df_dataset = pd.DataFrame()

    for subject in list_subjects:
        if "s" in subject:
            print("Processing subject_id {}".format(subject[1:]))
            df_tmp = process_subject(os.path.join(args.data_dir, subject), int(subject[1:]))
            df_dataset = df_dataset.append(df_tmp)

            print("\n**************************************************************\n")

    df_dataset['azimuth'] = df_dataset['azimuth'].astype(int)

    # Map the azimuth angles to class labels between 0-N
    df_dataset.loc[df_dataset.azimuth < -90, "azimuth"] = -90
    df_dataset.loc[df_dataset.azimuth >= 90, "azimuth"] = 89
    df_dataset['azimuth'] = df_dataset['azimuth'] + 90
    # df_dataset['labels'] = df_dataset['labels'] // args.azimuth_resolution

    df_dataset.to_csv("{}.csv".format(args.output_filename), index=False)

    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Format data of several subjects (audioSamples, head angles, head joint) into a unified dataset")
    parser.add_argument("data_dir", type=str,
                        help="Path directory of audio samples")

    parser.add_argument("--output_filename", type=str, default="output_dataset",
                        help="Output filename")

    parser.add_argument("--azimuth_resolution", type=int, default=1,
                        help="Angle resolution for azimuth")

    parser.add_argument("--elevation_resolution", type=int, default=1,
                        help="Angle resolution for elevation")

    parser.add_argument("--time_window", type=int, default=None,
                        help="length of the chunk audio")

    parser.add_argument("--dataset", type=str, default=None,
                        help="length of the chunk audio")

    parser.add_argument("--save_path", type=str, default='/tmp',
                        help="length of the chunk audio")

    parser.add_argument("--gammatone", help="Enable gammatone filter processing", action="store_true")

    parser.add_argument("--augment", help="Enable data augmentation", action="store_true")

    parser_args = parser.parse_args()

    if parser_args.time_window and parser_args.dataset:
        df = pd.read_csv(parser_args.dataset)
        output_dir = os.path.join(parser_args.save_path, f'dataset-{parser_args.time_window}')
        df = create_chunk_audio(df, output_dir, parser_args.time_window)

    elif parser_args.augment:
        df = pd.read_csv(parser_args.dataset)
        output_dir = os.path.join(parser_args.save_path, f'dataset-augmented')

        res = data_augmentation(df, output_dir, PATH_DATA)
        res.to_csv("dataset_augmented.csv", index=False)

    elif parser_args.gammatone and parser_args.dataset:
        df = pd.read_csv(parser_args.dataset)
        num_cores = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=num_cores)
        df_filename = df['audio_filename']

        apply_by_multiprocessing(df_filename, create_gammatone, workers=num_cores)

    else:
        main(parser_args)

    sys.exit()
