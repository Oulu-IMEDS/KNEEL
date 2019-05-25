import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--oai_data_root', default='')
    parser.add_argument('--oai_metadata_root', default='')
    parser.add_argument('--oai_landmarks_root', default='')
    args = parser.parse_args()
    # low-budget and high budget annotations must be stored in the same file.