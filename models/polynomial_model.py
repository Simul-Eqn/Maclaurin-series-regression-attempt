
import torch 
import torch.nn as nn 

import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class PolynomialModel(nn.Module): 
    def __init__(self, degree=4, device=device): 
        super(PolynomialModel, self).__init__() 
        self.degree = degree 
        self.device = device 
        self.init_weights() 

    def init_weights(self): 
        w = nn.init.uniform_(torch.empty(self.degree+1), -1/np.sqrt(self.degree+1), 1/np.sqrt(self.degree+1)) 
        self.weights = [nn.Parameter(w[i]) for i in range(self.degree+1)] 
    
    def forward(self, x:torch.Tensor): 
        # get x values for many powers 
        xpow = x.repeat(self.degree+1,1).pow(torch.arange(self.degree+1, dtype=torch.float, requires_grad=True, device=self.device).repeat_interleave(x.shape[0]).reshape(self.degree+1,-1)) 
        return torch.matmul(xpow.transpose(0, 1), torch.stack(self.weights,dim=0)) # then multiply by weights, reduce dimension back to 1. 
        #return self.layer(xpow.transpose(0,1)) 
        
    def scaling_to_lrs(self, scaling, base_lr:float): 
        res = [] 
        for i in range(len(scaling)): 
            res.append({'params': self.weights[i], 'lr':base_lr*scaling[i]}) 
        return res 


# instead, i want to specifically make higher powers' loss consider far from centre more, lower powers' loss consider more normal or opposite of far from center 
# i do this by varying learning rate in the optimizer. 
