import os 

import re 

from losses import * 
from models import * 
import params 

import numpy as np

import torch 
import torch.nn as nn 

torch.manual_seed(params.seed) 
torch.cuda.manual_seed(params.seed) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

sigmoid = nn.Sigmoid() 
def f(x): 
    return np.sin(x)-sigmoid(x) 

target_xs = torch.linspace(-0.5, 0.5, 100) 
target_ys = f(target_xs) 

test_xs = torch.linspace(-0.9, 0.9, 10) 
test_ys = f(test_xs) 



     


runs = [
     {'testname': "POLYNOMIAL MODEL WITH NORMAL MSE LOSS", 
     'dir': 'results/polynomial_model/normal_loss/0.01', 
     'model': PolynomialModel(params.degree, device=device), 
     'losstype': NormalLoss(), 
     'test_losses': [NormalLoss()], 
     'num_epochs': 100, 
     'batch_size': 5, 
     'test_epoch_interval': 10, 
     'lr': 0.01, 
     }, 

     {'testname': "POLYNOMIAL MODEL WITH ABS FFZ MSE LOSS", 
     'dir': 'results/polynomial_model/mse_ffz_loss/0.1(2^(-0.01epoch))', 
     'model': PolynomialModel(params.degree, device=device), 
     'losstype': FarFromCenterLoss(), 
     'test_losses': [NormalLoss()], 
     'num_epochs': 1500, 
     'batch_size': 5, 
     'lr_func': lambda epoch: 0.1*(2**(-0.01*epoch)), 
     },
] 



for run in runs: 
    keys = run.keys() 

    savedir = run['dir'] 
    path = re.split("\\|/", savedir) 
    curr = "" 
    for i in range(len(path)): 
        curr = os.path.join(curr, path[i]) 
        try: 
            os.mkdir(curr) 
        except: 
            # means it already exists 
            pass 
    
    if "filename" in keys: 
        filename = keys['filename'] 
    else: 
        filename = "results.txt" 

    outstr = open(os.path.join(savedir, filename), 'w') 

    outstr.write(run['testname']+"\n\n") 
    print("\nTESTING "+run['testname']) 
    
    model = run['model'] 
    losstype = run['losstype'] 
    test_losses = run['test_losses'] 

    if 'num_epochs' in keys: 
        num_epochs = run['num_epochs'] 
    else: 
        num_epochs = params.num_epochs 

    if 'batch_size' in keys: 
        batch_size = run['batch_size'] 
    else: 
        batch_size = params.batch_size 

    if 'test_epoch_interval' in keys: 
        test_epoch_interval = run['test_epoch_interval'] 
    else: 
        test_epoch_interval = params.test_epoch_interval 
    
    if 'lr' in keys: 
        if type(run['lr']) == type(1.0): 
            lr_func = lambda x: run['lr'] 
        else: 
            lr_func = run['lr'] 
    else: 
        if 'lr_func' in keys: 
            lr_func = run['lr_func']
        else: 
            lr_func = lambda x: params.lr 

    for epoch in range(num_epochs): 
        losses = [] 
        for batchstart in range(0, 100, batch_size): 
            pred = model(target_xs[batchstart:batchstart+batch_size]) 
            loss = losstype(pred.view(-1), target_ys[batchstart:batchstart+batch_size] 
                            + torch.rand(batch_size)/30) # and add noise hehe
            losses.append(loss.item()) 

            loss.backward() 

            optimizer = torch.optim.Adam(model.scaling_to_lrs(losstype.get_scalings(model.degree, target_xs[batchstart:batchstart+batch_size].mean(-1)), lr_func(epoch)), lr=lr_func(epoch))
            optimizer.step() 
            optimizer.zero_grad() 

            #print(list(normal_model.parameters()))
        
        outstr.write("EPOCH "+str(epoch)+" AVG LOSS: "+str(sum(losses)/len(losses))+'\n') 

        if ((epoch+1)%test_epoch_interval) == 0: 
            #print('TESTING')
            with (torch.no_grad()): 
                pred = model(test_xs) 
                outstr.write("XS: "+str(test_xs.tolist())+'\n')
                outstr.write("PRED: "+str(pred.view(-1).tolist())+'\n') 
                outstr.write("CORRECT: "+str(test_ys.tolist())+'\n') 

                for losstype in test_losses: 
                    loss = losstype(pred.view(-1), test_ys) 
                    outstr.write(str(losstype.__class__)+": "+str(loss.item())+'\n') 

    outstr.close() 




"""

# test far from zero loss 
print("\n\nABS FAR FROM ZERO MODEl: ")
ffz_model = PolynomialModel(params.degree, device=device) 
ffz_optimizer = torch.optim.SGD(ffz_model.parameters(), lr=params.lr)
for epoch in range(params.num_epochs): 
    losses = [] 
    for batchstart in range(0, 100, params.batch_size): 
        ffz_optimizer.zero_grad() 

        pred = ffz_model(target_xs[batchstart:batchstart+params.batch_size]) 
        targ_xs = target_xs[batchstart:batchstart+params.batch_size].view(params.batch_size,1) 
        targ_ys = target_ys[batchstart:batchstart+params.batch_size].view(params.batch_size,1) + torch.rand(params.batch_size,1)/30 # and add noise hehe
        loss = absffzloss(pred, targ_xs, targ_ys) 
        #print("EPOCH",epoch,"LOSS:",loss) 
        losses.append(loss.item())

        loss.backward() 
        ffz_optimizer.step() 

        #print(list(ffz_model.parameters()))
    
    print("EPOCH",epoch,"AVG LOSS:",sum(losses)/len(losses))

    if ((epoch+1)%params.test_epoch_interval) == 0: 
        with (torch.no_grad()): 
            pred = ffz_model(test_xs) 
            print("PRED:",pred.transpose(0,1)) 
            print("CORRECT:",test_ys) 
            loss = normalloss(pred, test_ys.view(10,1)) 
            print("NORMAL LOSS:", loss) 
            loss2 = absffzloss(pred, test_xs.view(10,1), test_ys.view(10,1)) 
            print("ABS FFZ LOSS:", loss2)


"""