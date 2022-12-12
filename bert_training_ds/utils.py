""" Utilities """
import argparse
import os
import csv

from transformers import BertForSequenceClassification, BertConfig

def rough_flops(h, s, n):
    """
        Compute a rough estimation of the FLOPS for fwd+bwd of a BERT layer. 
        - h: hidden size
        - s: sequence length
        - assuming the MLP is h x 4h x h
    """
    att, ffn = 4 * h * s**2 + 8 * s * h**2, 16 * s * h**2
    rough_fwd_flops = (att + ffn) * n
    rough_bwd_flops = rough_fwd_flops * 2 + rough_fwd_flops # assuming activation ckpt enabled
    return rough_fwd_flops, rough_bwd_flops

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True, help='Batch size. ')
    parser.add_argument('--hs', type=int, required=True, help='Hidden size. ')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads. ')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length. ')
    parser.add_argument('--csv_file', type=str, default='stats.csv', help='Path to the CSV file for storing results. ')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers. ')
    return parser

def build_model(args):
    """
        Build BERT model according to the specification. 
    """
    config = BertConfig(
        vocab_size=32003,
        hidden_size=args.hs,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hs*4,
        max_position_embeddings=args.seq_len,
        num_labels=10)
    model = BertForSequenceClassification(config)
    model.gradient_checkpointing_enable()
    model.train()
    model = model.half()
    param_cnt = sum(p.numel() for p in model.parameters())
    print('Total params: ', param_cnt)
    rough_fwd_flops, rough_bwd_flops = rough_flops(args.hs, args.seq_len, args.num_layers)
    print('Rough FLOPS per sample: fwd {}, bwd {}'.format(rough_fwd_flops, rough_bwd_flops))
    return model

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
