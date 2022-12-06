""" HuggingFace BERT training with fake data """

import argparse
import deepspeed
import json
import os
import time
import torch
import deepspeed.comm as dist

from utils import build_layer, rough_flops, write_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True, help='Batch size. ')
    parser.add_argument('--hs', type=int, required=True, help='Hidden size. ')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads. ')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length. ')
    parser.add_argument('--csv_file', type=str, default='stats.csv', help='Path to the CSV file for storing results. ')
    parser = deepspeed.add_config_arguments(parser)

    args, _ = parser.parse_known_args()
    return args

def train_multi_gpu_ds(args, layer_engine: deepspeed.DeepSpeedEngine, warmup=5, n_batches=10):
    if layer_engine.local_rank == 0:
        print('Transferring input and label to GPU... ')
    inputs = torch.rand((args.bs, args.seq_len, args.hs), dtype=torch.float16)
    inputs = inputs.to(layer_engine.local_rank)
    labels = torch.rand((args.bs, args.seq_len, args.hs), dtype=torch.float16)
    labels = labels.to(layer_engine.local_rank)
    if layer_engine.local_rank == 0:
        print('Warming up... ')
    for i in range(warmup):
        outputs = layer_engine(inputs)
        loss = (labels - outputs[0]).sum()
        layer_engine.backward(loss)
        layer_engine.step()

    if layer_engine.local_rank == 0:
        print('Running... ')
    start = time.time()
    for i in range(n_batches):
        outputs = layer_engine(inputs)
        loss = (labels - outputs[0]).sum()
        layer_engine.backward(loss)
        layer_engine.step()
    end = time.time()
    dist.log_summary()

    if layer_engine.local_rank == 0:
        print('Done!')
    if layer_engine.local_rank == 0:
        print('Max memory allocated in MBs: ', torch.cuda.max_memory_allocated() / (1024*1024))
        print('Total time in seconds: ', end - start)
    avg = (end - start) / n_batches
    rough_fwd_flops, rough_bwd_flops = rough_flops(args.hs, args.seq_len)
    res_dict = {
        'batch_size': args.bs,
        'seq_len': args.seq_len,
        'hidden_size': args.hs,
        'num_heads': args.num_heads,
        'avg_time_ms': avg * 1e3,
        'avg_TFLOPS': (rough_bwd_flops + rough_fwd_flops) * args.bs / (avg * 1e12) 
    }
    return res_dict

if __name__ == '__main__':
    # Parse args
    args = parse_args()
    deepspeed_config = json.load(
        open(args.deepspeed_config, 'r', encoding='utf-8'))

    # Initialize deepspeed env
    deepspeed.init_distributed(dist_backend='nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])

    layer = build_layer(args)
    trainable_params = filter(lambda p: p.requires_grad, layer.parameters())

    model_engine, _, _, _ = deepspeed.initialize(
        args=args, 
        model=layer,
        model_parameters=trainable_params,
    )

    res_dict = train_multi_gpu_ds(args, model_engine)
    if model_engine.local_rank == 0:
        write_csv(res_dict, args.csv_file)
        print('Test done!')
    