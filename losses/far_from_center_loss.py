
import torch 
import torch.nn as nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class FarFromCenterLoss(nn.Module): 
    def __init__(self, scaling_func=None, loss_func=nn.MSELoss(), device=device): 
        super(FarFromCenterLoss, self).__init__() 
        if scaling_func == None: 
            def scaling_func(degree, diff): 
                return torch.abs(diff)*(degree**2) 
        # scaling func takes in distance from 
        self.scaling_func = scaling_func 
        self.loss_func = loss_func 
        self.device = device 

    def get_scalings(self, max_degree, targetx, center=0.0): 
        return [self.scaling_func(degree, targetx-center) for degree in range(max_degree)] 
    
    def forward(self, resulty, targety): 
        #print("SCALING:",self.scaling_func(targetx.to(self.device)-center))
        #print("LOSS:",self.loss_func(targety.to(self.device),resulty.to(self.device), self.scaling_func(targetx.to(self.device)-center)))
        return self.loss_func(targety.to(self.device),resulty.to(self.device)) 

# oddly, this may not really work since it's supposed to update the higher power weights more when further from center, and the lower power weights more than close to center, but idk 

