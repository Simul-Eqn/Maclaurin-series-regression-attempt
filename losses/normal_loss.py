
import torch 
import torch.nn as nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class NormalLoss(nn.Module): 
    def __init__(self, loss_func=nn.MSELoss(), device=device): 
        super(NormalLoss, self).__init__() 
        # scaling func takes in distance from 
        self.loss_func = loss_func 
        self.device = device 

    def get_scalings(self, max_degree, targetx, center=0.0): 
        return [1 for _ in range(max_degree)] 
    
    def forward(self, resulty, targety): 
        #print("SCALING:",self.scaling_func(targetx.to(self.device)-center))
        #print("LOSS:",self.loss_func(targety.to(self.device),resulty.to(self.device), self.scaling_func(targetx.to(self.device)-center)))
        return self.loss_func(targety.to(self.device),resulty.to(self.device)) 

# oddly, this may not really work since it's supposed to update the higher power weights more when further from center, and the lower power weights more than close to center, but idk 

