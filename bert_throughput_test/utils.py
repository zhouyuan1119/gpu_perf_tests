""" Utilities """
import argparse
import csv
import os
import torch.nn as nn

from transformers import BertLayer, BertConfig

class SequentialBertLayers(nn.Module):
    """ Wrapper around a number of BERT layers """
    def __init__(self, config, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(num_layers)])
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)[0]
        return x

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True, help='Batch size. ')
    parser.add_argument('--hs', type=int, required=True, help='Hidden size. ')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads. ')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length. ')
    parser.add_argument('--csv_file', type=str, default='stats.csv', help='Path to the CSV file for storing results. ')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers. ')
    return parser

def rough_flops(h, s, num_layers=1):
    """
        Compute a rough estimation of the FLOPS for fwd+bwd of a BERT layer. 
        - h: hidden size
        - s: sequence length
        - assuming the MLP is h x 4h x h
    """
    att, ffn = 4 * h * s**2 + 8 * s * h**2, 16 * s * h**2
    rough_fwd_flops = (att + ffn) * num_layers
    rough_bwd_flops = rough_fwd_flops * 2
    return rough_fwd_flops, rough_bwd_flops

def build_layers(args):
    """
        Build one or more BERT layers according to the specification. 
    """
    config = BertConfig(
        vocab_size=30522,
        hidden_size=args.hs,
        num_hidden_layers=1,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hs*4,
        max_position_embeddings=args.seq_len,
        num_labels=10)
    layers = SequentialBertLayers(config, args.num_layers)
    layers.train()
    layers = layers.half()
    param_cnt = sum(p.numel() for p in layers.parameters())
    print('Total params: ', param_cnt)
    rough_fwd_flops, rough_bwd_flops = rough_flops(args.hs, args.seq_len, args.num_layers)
    print('Rough FLOPS per sample: fwd {}, bwd {}'.format(rough_fwd_flops, rough_bwd_flops))
    return layers

def write_csv(res_dict, csv_file='stats.csv'):
    """
        Write the profiling results to CSV. 
    """
    write_header = not os.path.exists(csv_file)
    field_names = [
        'batch_size', 
        'seq_len',
        'hidden_size', 
        'num_heads', 
        'num_layers',
        'avg_time_ms',
        'avg_TFLOPS'
    ]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        writer.writerow(res_dict)
