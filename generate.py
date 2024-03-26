import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math, copy
import os
from model import *



class Tokenizer:
    def __init__(self,filename="./data/shakespearetrain.txt"):
        self.all_characters=""
        with open(filename) as filenar:
           READEN_TEXT = filenar.read() 
        self.all_characters = sorted(list(set(READEN_TEXT)))
        self.nbcharacters=len(self.all_characters)
    def encode(self,text):
        codedtext = np.zeros(len(text),dtype=np.int_)
        for c in range(len(codedtext)):
            codedtext[c] = self.all_characters.index(text[c])
        return codedtext

def generate(model, tokenizer, start="The king!", max_len=40, block_size=100,temperature=1):
    model.eval()
    simutext = start
    inputs=torch.tensor(tokenizer.encode(start))
    inputs=inputs.to(device)
    if len(inputs)>block_size:
        inputs = inputs[-block_size:]
    with torch.no_grad():
        for i in range(max_len):
            mask = subsequent_mask(len(inputs))
            mask = mask.to(device)
            outputs = model(inputs.unsqueeze(0),mask.unsqueeze(0)) 
            outputs = outputs[:,-1,:]
            outputs_dist = outputs.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(outputs_dist, 1)
            predicted_char = tokenizer.all_characters[top_i[0]]
            simutext += predicted_char
            inputs = torch.tensor(tokenizer.encode(simutext)).to(device)
            if len(inputs)>block_size:
                inputs = inputs[-block_size:]
    return simutext

checkpoint = torch.load('./checkpoint/ckpt.t7')
V=checkpoint['V']
d_model=checkpoint['d_model']
N = checkpoint['N']
d_ff = checkpoint['d_ff']
h = checkpoint['h']
tokenizer = Tokenizer()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size=256
model = make_model(block_size,V,N=N,d_model=d_model, d_ff=d_ff, h=h)
model=model.to(device)
model.load_state_dict(checkpoint['model'])

print(generate(model,tokenizer,max_len=10000,block_size=256,temperature=0.8))
