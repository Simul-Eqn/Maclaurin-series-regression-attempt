
import torch 
import torch.nn as nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class PolynomialModel(nn.Module): 
    def __init__(self, degree=4, device=device): 
        super(PolynomialModel, self).__init__() 
        self.degree = degree 
        self.layer = nn.Linear(degree+1, 1) 
    
    def forward(self, x:torch.Tensor): 
        # get x values for many powers 
        xpow = x.repeat(self.degree+1,1).pow(torch.arange(self.degree+1).repeat_interleave(x.shape[0]).reshape(self.degree+1,-1)) 
        #return torch.matmul(xpow.transpose(0, 1), self.weights) # then multiply by weights, reduce dimension back to 1. 
        return self.layer(xpow.transpose(0,1)) 


