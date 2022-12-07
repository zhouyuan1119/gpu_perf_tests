import argparse
import time
import torch
import torch.nn as nn
from utils import rough_flops, build_layers, write_csv, get_parser

def parse_args():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    return args

def train_single_gpu(args, layer: nn.Module, warmup=5, n_batches=100):
    print('Transfering layer to GPU... ')
    layer = layer.cuda()
    inputs = torch.rand((args.bs, args.seq_len, args.hs), dtype=torch.float16)
    inputs = inputs.cuda()
    labels = torch.rand((args.bs, args.seq_len, args.hs), dtype=torch.float16)
    labels = labels.cuda()
    print('Warming up... ')
    for i in range(warmup):
        outputs = layer(inputs)
        loss = (labels - outputs[0]).sum()
        loss.backward()

    print('Running... ')
    start = time.time()
    for i in range(n_batches):
        outputs = layer(inputs)
        loss = (labels - outputs[0]).sum()
        loss.backward()
    end = time.time()

    print('Done!')
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
    args = parse_args()
    layer = build_layers(args)
    res_dict = train_single_gpu(args, layer)
    write_csv(res_dict, args.csv_file)
