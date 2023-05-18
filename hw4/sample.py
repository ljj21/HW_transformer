# ========================================================
#             Media and Cognition
#             Homework 4  Sequence Modeling
#             sample.py - main program for text generation
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================

"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import argparse

def sample(start, num_samples, max_new_tokens, model_name, ckpt_path, data_root, device):
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    # model
    dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=ptdtype)
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(ckpt_path, 'latest.pth')
    print("sample from %s"%ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig[model_name]
    if 'model_args' in checkpoint:
        gptconf = checkpoint['model_args']
    model = GPT(**gptconf)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    print(data_root)
    meta_path = os.path.join(data_root, 'meta.pkl')
    
    print(f"Loading meta from {meta_path}...")
    print(meta_path)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')

if __name__ == '__main__':

    # set random seed for reproducibility
    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # set configurations of the model and sampling process
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='\n', help='start of the sample, "\n" or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"')
    parser.add_argument('--num_samples', type=int, default='10', help='the number of samples')
    parser.add_argument('--max_new_tokens', type=int, default='500', help='the number of tokens generated in each sample')
    parser.add_argument('--model_name', type=str, default='gpt2', help='name of the model')
    parser.add_argument('--ckpt_path', type=str, default='work_dirs/quansongci', help='path to load checkpoints')
    parser.add_argument('--data_root', type=str, default='data/quansongci', help='file of training and validation data')
    parser.add_argument('--device', type=str, help='cpu or cuda')

    opt = parser.parse_args()
    if opt.device is None:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    sample(opt.start, opt.num_samples, opt.max_new_tokens, opt.model_name, opt.ckpt_path, opt.data_root, opt.device)
