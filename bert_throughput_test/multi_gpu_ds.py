""" HuggingFace BERT training with fake data """

import argparse
import deepspeed
import json
import os
import time
import torch
import deepspeed.comm as dist

from utils import build_layers, rough_flops, write_csv, get_parser

def parse_args():
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)

    args, _ = parser.parse_known_args()

    return args

def train_multi_gpu_ds(args, model_engine: deepspeed.DeepSpeedEngine, warmup=5, n_batches=10):
    if model_engine.local_rank == 0:
        print('Transferring input and label to GPU... ')
    inputs = torch.rand((args.bs, args.seq_len, args.hs), dtype=torch.float16)
    inputs = inputs.to(model_engine.local_rank)
    labels = torch.rand((args.bs, args.seq_len, args.hs), dtype=torch.float16)
    labels = labels.to(model_engine.local_rank)
    if model_engine.local_rank == 0:
        print('Warming up... ')
    for i in range(warmup):
        outputs = model_engine(inputs)
        loss = (labels - outputs[0]).sum()
        model_engine.backward(loss)
        model_engine.step()

    if model_engine.local_rank == 0:
        print('Running... ')
    start = time.time()
    for i in range(n_batches):
        outputs = model_engine(inputs)
        loss = (labels - outputs[0]).sum()
        model_engine.backward(loss)
        model_engine.step()
    end = time.time()
    dist.log_summary()

    if model_engine.local_rank == 0:
        print('Done!')
    if model_engine.local_rank == 0:
        print('Max memory allocated in MBs: ', torch.cuda.max_memory_allocated() / (1024*1024))
        print('Total time in seconds: ', end - start)
    avg = (end - start) / n_batches
    rough_fwd_flops, rough_bwd_flops = rough_flops(args.hs, args.seq_len, args.num_layers)
    res_dict = {
        'batch_size': args.bs,
        'seq_len': args.seq_len,
        'hidden_size': args.hs,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'avg_time_ms': avg * 1e3,
        'avg_TFLOPS': (rough_bwd_flops + rough_fwd_flops) * args.bs / (avg * 1e12) 
    }
    return res_dict

if __name__ == '__main__':
    # Parse args
    args = parse_args()
    deepspeed_config = json.load(
        open(args.deepspeed_config, 'r', encoding='utf-8'))
    # Fix random seed for all processes
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Initialize deepspeed env
    deepspeed.init_distributed(dist_backend='nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])

    layers = build_layers(args)
    trainable_params = filter(lambda p: p.requires_grad, layers.parameters())

    model_engine, _, _, _ = deepspeed.initialize(
        args=args, 
        model=layers,
        model_parameters=trainable_params,
    )

    res_dict = train_multi_gpu_ds(args, model_engine)
    if model_engine.local_rank == 0:
        write_csv(res_dict, args.csv_file)
        print('Test done!')
    