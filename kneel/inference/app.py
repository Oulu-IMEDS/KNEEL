"""
    This micro-service takes a dicom image in and returns JSON with localized landmark coordinates.
    (c) Aleksei Tiulpin, University of Oulu, 2019
"""
import argparse
from flask import jsonify
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
import logging
from logging import handlers
from kneel.inference.pipeline import KneeAnnotatorPipeline
import base64
app = Flask(__name__)


@app.route('/kneel/predict/bilateral', methods=['POST'])
def analyze_knee():
    logger = logging.getLogger(f'kneel-backend:app')
    logger.info('Received DICOM')
    dicom_base64 = request.get_json(force=True)['dicom']
    dicom_binary = base64.b64decode(dicom_base64)
    raw = DicomBytesIO(dicom_binary)
    data = dcmread(raw)
    logger.info('DICOM read')
    landmarks = annotator.predict(data, args.roi_size_mm, args.pad, args.refine).squeeze()
    logger.info('Prediction successful')
    if landmarks is not None:
        res = {'R': landmarks[0].tolist(), 'L': landmarks[1].tolist(), }
    else:
        res = {'R': None, 'L': None}
    logger.info('Sending results back to the user')
    return jsonify(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lc_snapshot_path', default='')
    parser.add_argument('--hc_snapshot_path', default='')
    parser.add_argument('--roi_size_mm', type=int, default=140)
    parser.add_argument('--pad', type=int, default=300)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--refine', type=bool, default=False)
    parser.add_argument('--mean_std_path', default='')
    parser.add_argument('--addr', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--deploy', type=bool, default=False)
    parser.add_argument('--jit_trace', type=bool, default=False)
    parser.add_argument('--logs', default='/tmp/kneel.log')
    args = parser.parse_args()

    logging.basicConfig(filename=args.logs, filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    logger = logging.getLogger(f'kneel-backend:app')

    annotator = KneeAnnotatorPipeline(args.lc_snapshot_path, args.hc_snapshot_path,
                                      args.mean_std_path, args.device, jit_trace=args.jit_trace)

    if args.deploy:
        http_server = WSGIServer((args.addr, args.port), app, log=logger)
        logger.log(logging.INFO, f'Production server is running @ {args.addr}:{args.port}')
        http_server.serve_forever()
    else:
        logger.log(logging.INFO, f'Debug server is running @ {args.addr}:{args.port}')
        app.run(host=args.addr, port=args.port, debug=True)
