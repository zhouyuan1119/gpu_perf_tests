""" Utilities """
import csv
import os

from transformers import BertLayer, BertConfig

def rough_flops(h, s):
    """
        Compute a rough estimation of the FLOPS for fwd+bwd of a BERT layer. 
        - h: hidden size
        - s: sequence length
        - assuming the MLP is h x 4h x h
    """
    att, ffn = 4 * h * s**2 + 8 * s * h**2, 16 * s * h**2
    rough_fwd_flops = att + ffn
    rough_bwd_flops = rough_fwd_flops * 2
    return rough_fwd_flops, rough_bwd_flops

def build_layer(args):
    """
        Build one BERT layer according to the specification. 
    """
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
    rough_fwd_flops, rough_bwd_flops = rough_flops(args.hs, args.seq_len)
    print('Rough FLOPS per sample: fwd {}, bwd {}'.format(rough_fwd_flops, rough_bwd_flops))
    return layer

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
        'avg_time_ms',
        'avg_TFLOPS'
    ]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        writer.writerow(res_dict)
