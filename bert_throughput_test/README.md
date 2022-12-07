# BERT Training Throughput Testing
This folder contains some python scripts for benchmarking BERT training throughput on single GPU or 
multi-GPU with [DeepSpeed](https://www.deepspeed.ai). We use the vanilla BERT implementation from 
[HuggingFace](https://huggingface.co) for convenience. Use `single_gpu.py` for single-GPU experiments, 
and use `multi_gpu_ds.py` for multi-GPU experiments. See the help messages for command line arguments. 
This can also serve as a demo for the simplest version of single-machine multi-GPU training with 
DeepSpeed. 

The purpose of this test is to evaluate the training throughput (in TFLOPS per GPU) under various
batch sizes, layer sizes, sequence lengths, and ZeRO optimization levels. As a result, this test
simplifies normal BERT training in the following ways:
- We only consider BERT layers and ignore all other things in BERT models. 
- In each batch we all use the same input data and label that are transferred to the GPU beforehand. 
  This eliminates any possible overhead caused by inefficient data loaders, but may also make the 
  result optimistic. 
- We use a dummy loss function that simply subtracts the output and label then sums the result. 
