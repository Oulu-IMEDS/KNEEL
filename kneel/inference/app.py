from flask import Flask, request
import argparse
from kneel.inference.pipeline import KneeAnnotatorPipeline

app = Flask(__name__)


@app.route('/', methods=['POST'])
def analyze_knee():
    data = request.data
    return "Hello World!"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lc_snapshot_path', default='')
    parser.add_argument('--hc_snapshot_path', default='')
    parser.add_argument('--roi_size_mm', type=int, default=140)
    parser.add_argument('--pad', type=int, default=300)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--refine', type=bool, default=False)
    parser.add_argument('--mean_std_path', default='')
    args = parser.parse_args()

    annotator = KneeAnnotatorPipeline(args.lc_snapshot_path, args.hc_snapshot_path,
                                      args.mean_std_path, args.device)

    app.run()
