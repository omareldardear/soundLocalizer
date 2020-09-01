import glob

import librosa
import pandas as pd
import argparse
import sys, os
from utils import split_audio_chunks, filter_voice, ToolGammatoneFb, gcc_gammatoneFilter, butter_lowpass_filter
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm
import multiprocessing

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from CONFIG import *
import pickle


COLUMNS_ANGLES = ['index', 'timestamp', 'azimuth', 'elevation', 'vergence']
COLUMNS_JOINTS = ['index', 'timestamp', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5']
COLUMNS_DATA = ['subject_id', 'audio_filename', 'azimuth', 'elevation', 'joint0', 'joint1', 'joint2', 'joint3',
                'joint4', 'joint5']


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
        sample_rate, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename, size_chunks=length_audio)

        for j, (signal1, signal2) in enumerate(zip(chunks_channel1, chunks_channel2)):
            filename = str(index) + '-' + str(j) + '_' + str(item['subject_id']) + '.wav'
            filename_write = os.path.join(output_dir, filename)

            data = np.stack((signal1, signal2), axis=1)

            if not filter_voice(signal1, sample_rate):
                print(f"{filename} not containing voice")

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


def process_subject(path_subject, subject_id):
    audio_samples_list = glob.glob(os.path.join(path_subject, "2020*/*.wav"))
    angles_filename = glob.glob(os.path.join(path_subject, "sound*/logAngles/data.log"))[0]
    head_state_filename = glob.glob(os.path.join(path_subject, "sound*/logHeadState/data.log"))[0]

    df_head_angles = pd.read_csv(angles_filename, sep=' ', names=COLUMNS_ANGLES)
    df_head_joints = pd.read_csv(head_state_filename, sep=' ', names=COLUMNS_JOINTS)

    # Drop unnecessary columns
    df_head_angles = df_head_angles.drop(['index', 'vergence'], axis=1)
    df_head_joints = df_head_joints.drop(['index'], axis=1)

    print("Found {} audio samples ".format(len(audio_samples_list)))

    df_data = pd.DataFrame()

    for index, file_name in enumerate(audio_samples_list):
        tmp = file_name.split('/')[-1]
        start_timestamp = tmp.split('.wav')[0].split('_')[0]
        stop_timestamp = tmp.split('.wav')[0].split('_')[-1]

        index_angle = index
        index_joint = abs(df_head_joints['timestamp'] - float(start_timestamp)).idxmin()

        data_angle = df_head_angles.iloc[[index_angle], 1:3]
        data_joint = df_head_joints.iloc[[index_joint], 1:]

        data_angle.insert(0, 'subject_id', str(subject_id))
        data_angle.insert(1, 'audio_filename', str(file_name))

        result = pd.concat([data_angle.reset_index(drop=True), data_joint.reset_index(drop=True)], axis=1,
                           ignore_index=True)

        df_data = df_data.append(result)

    df_data.reset_index(drop=True, inplace=True)
    df_data.columns = COLUMNS_DATA

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
    df_dataset['elevation'] = df_dataset['elevation'].astype(int)

    # Map the azimuth angles to class labels between 0-N
    df_dataset['labels'] = df_dataset['azimuth'] + 90
    df_dataset['labels'] = df_dataset['labels'] // args.azimuth_resolution

    df_dataset.to_csv("{}.csv".format(args.output_filename), index=False)

    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Format data of several subjects (audioSamples, head angles, head joint) into a unified dataset ")
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

    parser_args = parser.parse_args()

    if parser_args.time_window and parser_args.dataset:
        df = pd.read_csv(parser_args.dataset)
        output_dir = os.path.join(parser_args.save_path, f'dataset-{parser_args.time_window}')
        df = create_chunk_audio(df, output_dir, parser_args.time_window)

    elif parser_args.gammatone and parser_args.dataset:
        df = pd.read_csv(parser_args.dataset)
        num_cores = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=num_cores)
        df_filename = df['audio_filename']

        apply_by_multiprocessing(df_filename, create_gammatone, workers=num_cores)

    else:
        main(parser_args)

    sys.exit()
