# Whole BERT Model Training with DeepSpeed
This folder contains some python scripts for benchmarking BERT training throughput on multi-GPU with 
[DeepSpeed](https://www.deepspeed.ai). We use the vanilla BERT implementation from 
[HuggingFace](https://huggingface.co) for convenience. Run this example with 
``` deepspeed train.py <args> ```. See the help messages for choice of arguments. 

Compared with `bert_throughput_test`, this test is more realistic in the sense that it trains the
whole BERT model. We still use fake training data, but this time we use DeepSpeed's data loader to
send every batch onto the GPU. Using real data should not make much of a difference. 