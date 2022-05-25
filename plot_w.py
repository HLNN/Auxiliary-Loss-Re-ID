import re

import matplotlib.pyplot as plt
import argparse


def main(args):
    with open(args.path, 'r') as f:
        log_text = f.read()
    w = re.findall(r'W: (\d.\d{3})\n', log_text)
    w = list(map(float, w))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(range(len(w)), w)
    ax.set_ylim(0., 2.)
    plt.show()



    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Weight")
    parser.add_argument("--path", required=True, help="path to log file", type=str)
    args = parser.parse_args()

    main(args)
