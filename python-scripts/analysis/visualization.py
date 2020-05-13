import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse



def main(args):
    df = pd.read_csv(args.data_filename)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=df['azimuth'],
                         y=df['elevation'],
                         c='DarkBlue')

    ax.set_title("Angles position")
    ax.set_xlabel("Azimuth (in degree)")
    ax.set_ylabel("Elevation (in degree)")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple visualization script ")
    parser.add_argument("data_filename", type=str,
                        help="Path filename of data")

    args = parser.parse_args()

    main(args)
