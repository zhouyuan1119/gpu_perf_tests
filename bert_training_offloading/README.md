# Whole BERT Model Training with Offloading
This folder contains Python scripts for training HuggingFace BERT models using offloading-based
techniques. Here we try [PatrickStar](https://github.com/Tencent/PatrickStar/tree/master/examples)
from Tencent. The code in this folder is adapted from 
[this example](https://github.com/Tencent/PatrickStar/blob/master/examples/huggingface_bert.py). 

Notice that PatrickStar uses [PyTorch DDP](https://pytorch.org/docs/stable/distributed.html) for
parallelizing across multiple GPUs or multiple machines. To run this test, use
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py <args>
``` 