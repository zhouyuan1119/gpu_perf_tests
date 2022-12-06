import argparse
import csv
import os
import time
import torch
import torch.nn as nn
from transformers import BertLayer, BertConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True, help='Batch size. ')
    parser.add_argument('--hs', type=int, required=True, help='Hidden size. ')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads. ')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length. ')
    args, _ = parser.parse_known_args()
    return args

def rough_flops(args):
    h, s= args.hs, args.seq_len
    att, ffn = 4 * h * s**2 + 8 * s * h**2, 16 * s * h**2
    rough_fwd_flops = att + ffn
    rough_bwd_flops = rough_fwd_flops * 2
    return rough_fwd_flops, rough_bwd_flops

def build_layer(args):
    config = BertConfig(
        vocab_size=30522,
        hidden_size=args.hs,
        num_hidden_layers=1,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hs*4,
        max_position_embeddings=args.seq_len,
        num_labels=10)
    layer = BertLayer(config)
    layer.train()
    layer = layer.half()
    param_cnt = sum(p.numel() for p in layer.parameters())
    print('Total params: ', param_cnt)
    rough_fwd_flops, rough_bwd_flops = rough_flops(args)
    print('Rough FLOPS per sample: fwd {}, bwd {}'.format(rough_fwd_flops, rough_bwd_flops))
    return layer

def train(args, layer: nn.Module, warmup=5, n_batches=100):
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
    rough_fwd_flops, rough_bwd_flops = rough_flops(args)
    res_dict = {
        'batch_size': args.bs,
        'seq_len': args.seq_len,
        'hidden_size': args.hs,
        'num_heads': args.num_heads,
        'avg_time_ms': avg * 1e3,
        'avg_TFLOPS': (rough_bwd_flops + rough_fwd_flops) * args.bs / (avg * 1e12) 
    }
    return res_dict

def write_csv(res_dict, csv_file='stats.csv'):
    write_header = not os.path.exists(csv_file)
    field_names = [
        'batch_size', 
        'seq_len',
        'hidden_size', 
        'num_heads', 
        'avg_time_ms',
        'avg_TFLOPS'
    ]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        writer.writerow(res_dict)

if __name__ == '__main__':
    args = parse_args()
    layer = build_layer(args)
    res_dict = train(args, layer)
    write_csv(res_dict)
