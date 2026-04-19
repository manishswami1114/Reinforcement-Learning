import torch 
import torch.nn as nn

class RiverRaid(nn.Module):
    def __init__(self,input_shape,n_actions):
        super(RiverRaid,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.conv(torch.zeros(1,*input_shape)).size()[-1]
            print(f"total linear input size is :{size}")
        self.fc = nn.Sequential(
            nn.Linear(size,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )
    def forward(self, x:torch.ByteTensor):
        xx = x/255
        return self.fc(self.conv(xx))