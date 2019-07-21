import pandas as pd
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kls_okoa', default='')
    parser.add_argument('--kls_maknee', default='')
    args = parser.parse_args()

    okoa = pd.read_csv(args.kls_okoa).drop('ID', 1)
    maknee = pd.read_csv(args.kls_maknee).drop('ID', 1)

    for ds, ds_name in zip([okoa, maknee], ['OKOA', 'MAKNEE']):
        print(f'==> {ds_name}')
        print(pd.isna(ds).sum())
        for KL in range(5):
            print(KL, np.sum(ds.values == KL))