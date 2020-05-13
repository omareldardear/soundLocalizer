import glob
import pandas as pd
import argparse
import sys, os

COLUMNS_ANGLES = ['index', 'timestamp', 'azimuth', 'elevation', 'vergence']
COLUMNS_JOINTS = ['index', 'timestamp', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5']
COLUMNS_DATA = ['audio_filename', 'azimuth', 'elevation', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5']


def main(args):
    audio_samples_list = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    df_head_angles = pd.read_csv(args.head_angle_filename, sep=' ', names=COLUMNS_ANGLES)
    df_head_joints = pd.read_csv(args.head_joint_filename, sep=' ', names=COLUMNS_JOINTS)

    # Drop unnessary columns
    df_head_angles = df_head_angles.drop(['index', 'vergence'], axis=1)
    df_head_joints = df_head_joints.drop(['index'], axis=1)

    print("Found {} audio samples ".format(len(audio_samples_list)))

    df_data = pd.DataFrame()

    for sample in audio_samples_list:
        file_name = sample.split('/')[-1]
        stop_timestamp = file_name.split('.wav')[0].split('_')[-1]

        index_angle = abs(df_head_angles['timestamp'] - float(stop_timestamp)).idxmin()
        index_joint = abs(df_head_joints['timestamp'] - float(stop_timestamp)).idxmin()

        data_angle = df_head_angles.iloc[[index_angle], 1:3]
        data_joint = df_head_joints.iloc[[index_joint], 1:]

        data_angle.insert(0, 'audio_filename', file_name)

        result = pd.concat([data_angle.reset_index(drop=True), data_joint.reset_index(drop=True)], axis=1, ignore_index=True)

        df_data = df_data.append(result)

    df_data.reset_index(drop=True, inplace=True)
    df_data.columns = COLUMNS_DATA
    df_data.to_csv("soundLocalisation_dataset.csv", index=False)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Format data (audioSamples, head angles, head joint) into a unified dataset ")
    parser.add_argument("audio_dir", type=str,
                        help="Path directory of audio samples")
    parser.add_argument("head_angle_filename", type=str,
                        help="Path of the head angles data")
    parser.add_argument("head_joint_filename", type=str,
                        help="Path of the head joint data")
    args = parser.parse_args()

    main(args)

    sys.exit()
