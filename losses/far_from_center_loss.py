
import torch 
import torch.nn as nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class FarFromCenterLoss(nn.Module): 
    def __init__(self, scaling_func=lambda x: torch.abs(x), loss_func=None, device=device): 
        super(FarFromCenterLoss, self).__init__() 
        if loss_func==None: 
            def loss_func(input,target,weight): 
                return torch.sum(weight * (input - target) ** 2)
        # scaling func takes in distance from 
        self.scaling_func = scaling_func 
        self.loss_func = loss_func 
        self.device = device 
    
    def forward(self, resulty, targetx, targety, center=0.0): 
        #print("SCALING:",self.scaling_func(targetx.to(self.device)-center))
        #print("LOSS:",self.loss_func(targety.to(self.device),resulty.to(self.device), self.scaling_func(targetx.to(self.device)-center)))
        return self.loss_func(targety.to(self.device),resulty.to(self.device), self.scaling_func(targetx.to(self.device)-center)) 

# oddly, this may not really work since it's supposed to update the higher power weights more when further from center, and the lower power weights more than close to center, but idk 

