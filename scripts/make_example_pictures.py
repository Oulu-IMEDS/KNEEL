import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lc_annotations', default='')
    parser.add_argument('--hc_annotations', default='')
    parser.add_argument('--kls', default='')
    parser.add_argument('--lc_images', default='')
    parser.add_argument('--hc_images', default='')
    args = parser.parse_args()

