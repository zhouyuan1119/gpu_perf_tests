""" HuggingFace BERT training with fake data """

import argparse
import deepspeed
import json
import os
import time
import torch

import torch.nn as nn
import deepspeed.comm as dist

from fake_dataset import FakeDataset
from utils import build_model, get_parser, rough_flops, write_csv

def parse_args():
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)

    args, _ = parser.parse_known_args()

    return args

def train_multi_gpu_ds(args, model_engine: deepspeed.DeepSpeedEngine, train_loader, warmup=5, n_batches=10):
    if model_engine.local_rank == 0:
        print('Start training!')
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(train_loader):
        if i == warmup + n_batches:
            break
        if i == warmup:
            start = time.time()
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        outputs = model_engine(inputs).logits
        loss = criterion(outputs, labels)
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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Initialize deepspeed env
    deepspeed.init_distributed(dist_backend='nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize fake data, model and optimizer
    dataset = FakeDataset(
        (args.seq_len,), 
        50000, # 5k samples should be enough for our purpose
        torch.int64, 
        (0, 32003),
        10
    )
    model = build_model(args)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, _, train_loader, _ = deepspeed.initialize(
        args=args, 
        model=model,
        model_parameters=trainable_params,
        training_data=dataset
    )

    res_dict = train_multi_gpu_ds(args, model_engine, train_loader)
    if model_engine.local_rank == 0:
        write_csv(res_dict, args.csv_file)
        print('Test done!')
    