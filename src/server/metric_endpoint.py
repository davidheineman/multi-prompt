import os, sys

os.environ['NCCL_P2P_DISABLE'] = '1'

import torch
import torch.distributed as dist
from flask import Flask, request, jsonify

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from utils.distributed_inference import setup_model_parallel, generate_distributed, generate_distributed_child

METRIC_PORT = int(os.environ.get("METRIC_PORT", 8501))
METRIC_NAME = str(os.environ.get("METRIC_NAME", "no_metric"))

app = Flask(__name__)


def setup(metric_name):
    global multi_gpu_inference, local_rank, metric, metric_rf
    multi_gpu_inference = torch.distributed.is_available()

    if "LOCAL_RANK" in os.environ and multi_gpu_inference:
        local_rank, _ = setup_model_parallel()
        print(f'Loading model to device: cuda:{local_rank}')

    match metric_name:
        case 'lens':
            from src.metrics.simplification import Lens
            # Normally, batch_size=512 is good, but it fails on OLMo. So, we have to use batch_size=128
            metric    = Lens(variation='lens', batch_size=128, devices=[local_rank])
            metric_rf = Lens(variation='lens_salsa', batch_size=128, devices=[local_rank])
        case 'sle':
            from src.metrics.simplification import SLE
            metric_rf = SLE(devices=[local_rank])
        case 'bleu':
            from src.metrics.translation import BLEU
            metric    = BLEU()
        case 'comet':
            from src.metrics.translation import Comet
            metric    = Comet(variation='comet_22', batch_size=32, devices=[local_rank])
            metric_rf = Comet(variation='comet_kiwi_23', size='xl', batch_size=32, devices=[local_rank])
        case 'comet_kiwi_xxl':
            from src.metrics.translation import Comet
            metric_rf = Comet(variation='comet_kiwi_23', size='xxl', batch_size=8, devices=[local_rank])
        case 'xcomet':
            from src.metrics.translation import XComet
            metric    = XComet(size='xl', devices=[local_rank])
        case 'metricx':
            from src.metrics.translation import MetricX
            metric    = MetricX(variation='metricx', size='xl', devices=[local_rank])
        case 'metricx_qe':
            from src.metrics.translation import MetricX
            metric_rf = MetricX(variation='metricx_qe', size='xl', devices=[local_rank])
        case 'sari':
            from src.metrics.simplification import Sari
            metric    = Sari()
        case 'bert_score':
            from src.metrics.translation import BertScore
            metric    = BertScore(devices=[local_rank])
        case 'bart_score':
            from src.metrics.translation import BartScore
            metric    = BartScore(devices=[local_rank])
        case 'code_bert_score':
            from src.metrics.code import CodeBertScore
            metric    = CodeBertScore(devices=[local_rank])
        case 'mbr_exec':
            raise RuntimeError('MBR Exec endpoint not supported, as this is a GPU-free metric. It is calculated by the MBR processor instead.')
        case _: raise ValueError(f'Metric not supported: "{METRIC_NAME}".')


@app.route('/bleu_eval', methods=['POST'])
@app.route('/sari_eval', methods=['POST'])
@app.route('/lens_eval', methods=['POST'])
@app.route('/bert_score_eval', methods=['POST'])
@app.route('/bart_score_eval', methods=['POST'])
@app.route('/code_bert_score_eval', methods=['POST'])
@app.route('/comet_eval', methods=['POST'])
@app.route('/xcomet_eval', methods=['POST'])
@app.route('/metricx_eval', methods=['POST'])
def eval():
    final_scores = generate_distributed(request.json, metric)
    response = {'scores': final_scores}
    return jsonify(response)


@app.route('/lens_salsa_eval', methods=['POST'])
@app.route('/sle_eval', methods=['POST'])
@app.route('/comet_kiwi_eval', methods=['POST'])
@app.route('/comet_kiwi_xxl_eval', methods=['POST'])
@app.route('/metricx_qe_eval', methods=['POST'])
def eval_single_gpu():
    if int(os.environ.get("WORLD_SIZE", -1)) > 1:
        raise NotImplementedError(f"Multi-GPU, reference-free evaluation has not been implemented.")
    final_scores = generate_distributed(request.json, metric_rf)
    response = {'scores': final_scores}
    return jsonify(response)


@app.route('/ping')
def ping():
    return 'Pong!'


if __name__ == '__main__':
    setup(METRIC_NAME)

    if multi_gpu_inference:
        try:
            dist.get_rank()
        except RuntimeError as e:
            raise RuntimeError(f'Failed to initialize distributed setup. Did you mean to run `torchrun --nproc_per_node X {__name__}`? Threw: {e}')

    if not multi_gpu_inference or local_rank == 0:
        app.run(host='0.0.0.0', port=METRIC_PORT)
    else:
        while True:
            try:
                generate_distributed_child(metric)
            except Exception as e:
                print(f'Device cuda:{local_rank} threw exception: {e}')
                pass
