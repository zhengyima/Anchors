import time
import argparse
import pickle
import random
import numpy as np
import os
import json
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",
                    type=str)
parser.add_argument("--output_dir",
                    type=str)
parser.add_argument("--limit",
                    type=int, default=-1)
parser.add_argument("--shuffle",action="store_true")
args = parser.parse_args()

def make_shuffle():
    print("start shuffle.............")
    filenames = os.listdir(args.input_dir)
    datas = []
    for fn in tqdm(filenames, desc="files"):
        
        fpath = os.path.join(args.input_dir, fn)
        print(fpath + "read success................")
        with open(fpath,'r',encoding='utf-8') as f:
            for line in tqdm(f, desc=f"{fn} lines"):
                datas.append(line.strip())
    
    print("read success..................")
    newdatas = []
    
    len_data = len(datas)
    num = np.arange(len_data)

    if args.shuffle:
        np.random.shuffle(num)
    
    print("start shuffling...............")
    limit = len_data
    if args.limit > 0:
        limit = min(args.limit, len_data)
    if limit % 32 != 0:
        limit -= limit % 32

    with open(os.path.join(args.output_dir, "epoch_0.json"),'w',encoding='utf-8') as wf:
        for i in tqdm(range(limit)):
            wf.write(datas[i]+"\n")

    print("shuffle success...............")
    
if __name__ == '__main__':
    make_shuffle()
