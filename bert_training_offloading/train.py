""" HuggingFace BERT training with fake data """

import json
import time
import torch

import torch.nn as nn

from fake_dataset import FakeDataset
from utils import build_model, get_parser, rough_flops, write_csv
from patrickstar.runtime import initialize_engine
from patrickstar.utils import get_rank
from torch.utils.data import DataLoader


def parse_args():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    return args

def train_multi_gpu_pstar(args, model_engine, optim, train_loader, warmup=5, n_batches=10):
    local_rank = get_rank()
    device = torch.device(f"cuda:{local_rank}")
    if local_rank == 0:
        print('Start training!')
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(train_loader):
        if local_rank == 0:
            print('Batch ', i)
        if i == warmup + n_batches:
            break
        if i == warmup:
            start = time.time()
        optim.zero_grad()
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model_engine(inputs).logits
        loss = criterion(outputs, labels)
        model_engine.backward(loss)
        optim.step()
    end = time.time()

    if local_rank == 0:
        print('Done!')
    if local_rank == 0:
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
    config = json.load(open(args.config, 'r', encoding='utf-8'))
    # Fix random seed for all processes
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Initialize PatrickStar env
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(get_rank())

    # Initialize fake data, model and optimizer
    dataset = FakeDataset(
        (args.seq_len,), 
        50000, # 5k samples should be enough for our purpose
        torch.int64, 
        (0, 32003),
        10
    )
    train_loader = DataLoader(
        dataset, 
        batch_size=args.bs, 
        shuffle=True)

    # PatrickStar requires we wrap the model with a function
    def model_func():
        return build_model(args)

    model, optim = initialize_engine(
        model_func=model_func, local_rank=get_rank(), config=config
    )

    # Start training
    res_dict = train_multi_gpu_pstar(args, model, optim, train_loader)
    if get_rank() == 0:
        write_csv(res_dict, args.csv_file)
        print('Test done!')
    