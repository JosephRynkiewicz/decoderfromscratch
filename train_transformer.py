import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import math, copy, time
import matplotlib.pyplot as plt
import os
from model import *
from tqdm import tqdm
import random
 


##################################################################################
### Exploration
##################################################################################

#Character data

class CFG:
    num_workers=2
    device='cuda:0'


parameters = {
    'epochs': 10,
    'lr': 1e-3,
    'batch_size': 256,
    'weight_decay': 1e-9,
    'block_size': 256,
    'd_model': 512,
    'N': 6,
    'd_ff': 2048,
    'h':8,
    'drop_out': 0.5,
    'seed': 0,
    'lr': 1e-3,
}



def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False

fix_seed(parameters['seed'])  
    

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

    

class TextDataset(Dataset):
    def __init__(self,tokenizer,filename="./data/shakespearetrain.txt",block_size=256):
        super().__init__()
        self.mask = subsequent_mask(block_size)
        with open(filename) as filenar:
            READEN_TEXT = filenar.read()
        self.READEN_TOKENS = tokenizer.encode(READEN_TEXT)
        self.block_size=block_size
    def __len__(self):
        return len(self.READEN_TOKENS)-self.block_size-1
    def __getitem__(self,item):
        inputs_id= self.READEN_TOKENS[item:item+self.block_size+1]
        return inputs_id[:-1], self.mask, inputs_id[1:]





device = CFG.device
print("device : ",device)
block_size=parameters['block_size']
tokenizer = Tokenizer()
datatrain = TextDataset(tokenizer,filename="./data/shakespearetrain.txt",block_size=block_size)
datavalid = TextDataset(tokenizer,filename="./data/shakespearevalid.txt",block_size=block_size)
trainloader = DataLoader(datatrain,batch_size=256,shuffle=True,drop_last=True,num_workers=2)
validloader = DataLoader(datavalid,batch_size=256,shuffle=False,drop_last=True,num_workers=2) 
V = tokenizer.nbcharacters
d_model = parameters['d_model']
N = parameters['N']
d_ff = parameters['d_ff']
h = parameters['h']
model = make_model(block_size, V,N=N,d_model=d_model, d_ff=d_ff, h=h,dropout=parameters['drop_out'], device=device)
model=model.to(device);



lr = parameters['lr']
criterion = nn.NLLLoss()
wd = parameters['weight_decay']
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
lr_sc = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(trainloader), epochs=parameters['epochs'])


# Training
def train(epoch,trainloader, model, optimizer, lr_sc):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, masks, targets) in loop:
        inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs,masks)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        targets = targets.contiguous().view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        lr_sc.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(acc=correct/total)
    return correct/total

def valid(epoch, validloader, model):
    global best_acc
    model.eval()
    train_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(validloader), total=len(validloader))
        for batch_idx, (inputs, masks, targets) in loop:
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            outputs = model(inputs,masks)
            outputs = outputs[:,-1,:]
            targets = targets[:,-1]
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loop.set_description(f"Epoch [{epoch}]")
            loop.set_postfix(acc=correct/total)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'V': V,
            'd_model': d_model,
            'N': N,
            'd_ff': d_ff,
            'h': h,
            'model': model.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc


best_acc = 0    
for epoch in range(parameters['epochs']):
    acc_train = train(epoch,trainloader, model, optimizer, lr_sc)
    print("Accuracy train: ", acc_train)
    acc_valid = valid(epoch,validloader, model)
    print("Accuracy valid: ", acc_valid)




 




