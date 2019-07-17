import argparse
import pandas as pd
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lc_annotations', default='')
    parser.add_argument('--hc_annotations', default='')
    parser.add_argument('--kls', default='')
    parser.add_argument('--lc_images', default='')
    parser.add_argument('--hc_images', default='')
    args = parser.parse_args()

    kls = pd.read_csv(args.kls)
    lc_annotations = pd.read_csv(args.lc_annotations)
    hc_annotations = pd.read_csv(args.hc_annotations)

    hc_images = glob.glob(os.path.join(args.hc_images, '*'))
    lc_images = glob.glob(os.path.join(args.lc_images, '*'))
