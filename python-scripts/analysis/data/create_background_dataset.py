import glob

import librosa
import pandas as pd
import argparse
import sys, os

from scipy.io import wavfile

from utils import removeNoise, split_audio_chunks, filter_voice, ToolGammatoneFb, gcc_gammatoneFilter, butter_lowpass_filter, pitch_augment
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm
import multiprocessing


from CONFIG import *
import pickle
import random


def create_chunk_audio_background(audio_filename, output_dir, length_audio):
    """
    Create from a wav file chunk of length audio if they contain x% of voice and saved them
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Format data of background noise (wav file) "
                    " split them into chuncks define by time_window parameter")
    parser.add_argument("background_path", type=str,
                        help="Path  of audio file with background noise")

    parser.add_argument("--time_window", type=int, default=1000,
                        help="length of the chunk audio")

    parser_args = parser.parse_args()

    df = create_chunk_audio(parser_args.background_path, f"/tmp/background_dataset-{parser_args.time_window}", parser_args.time_window)

    print(f"Created background dataset with {df.shape[0]} rows")