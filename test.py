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


absffzloss = FarFromCenterLoss() # ffz because far from center but center is zero 
normalloss = nn.MSELoss() 

# test normal loss 
print("\n\nNORMAL MODEl: ")
normal_model = PolynomialModel(params.degree, device=device) 
normal_optimizer = torch.optim.SGD(normal_model.parameters(), lr=params.lr)
for epoch in range(params.num_epochs): 
    losses = [] 
    for batchstart in range(0, 100, params.batch_size): 
        normal_optimizer.zero_grad() 

        pred = normal_model(target_xs[batchstart:batchstart+params.batch_size]) 
        loss = normalloss(pred, target_ys[batchstart:batchstart+params.batch_size].view(params.batch_size,1)
                          + torch.rand(params.batch_size,1)/30) # and add noise hehe
        #print("EPOCH",epoch,"LOSS:",loss) 
        losses.append(loss.item()) 
        #print(loss.item())
        #print("PRED:",pred.transpose(0,1)) 
        #print("TARGET:",target_ys[batchstart:batchstart+params.batch_size]) 
        #print(pred.transpose(0,1)-target_ys[batchstart:batchstart+params.batch_size]) 

        loss.backward() 
        normal_optimizer.step() 

        #print(list(normal_model.parameters()))
    
    print("EPOCH",epoch,"AVG LOSS:",sum(losses)/len(losses))

    if ((epoch+1)%params.test_epoch_interval) == 0: 
        with (torch.no_grad()): 
            pred = normal_model(test_xs) 
            print("PRED:",pred.transpose(0,1)) 
            print("CORRECT:",test_ys) 
            loss = normalloss(pred, test_ys.view(10,1)) 
            print("NORMAL LOSS:", loss) 
            loss2 = absffzloss(pred, test_xs, test_ys.view(10,1)) 
            print("ABS FFZ LOSS:", loss2)


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


