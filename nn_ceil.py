import  torch
from torch import nn

input=torch. tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1,],
                     [2,1,0,1,1]],dtype=torch.float)
input=torch.reshape(input,(-1,1,5,5))

class HankMd(nn.Module):
    def __init__(self):
        super(HankMd,self).__init__()
        self.maxpool = nn.MaxPool2d(2,ceil_mode=True)
    def forward(self,x):
        x = self.maxpool(x)
        return x

model = HankMd()
output = model(input)
print(output)


