import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()

    def forward(self, x):
        output=x+1
        return output

# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [1,2,3,4,5]])
# kernel=torch.tensor([[1,2,1],
#                      [0,1,0],
#                      [2,1,1]])
# input=torch.reshape(input,(1,1,5,5))
# kernel=torch.reshape(kernel,(1,1,3,3))
# print(input)
# print(kernel)
# conv=F.conv2d(input,kernel,stride=1,padding=0)
# print(conv)
# conv=F.conv2d(input,kernel,stride=1,padding=1)
# print(conv)

dataset=torchvision.datasets.CIFAR10(root='./set01',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3)
    def forward(self,x):
        x=self.conv1(x)
        return x
writer=SummaryWriter('.idea/logs')
index=0
for data in dataloader:
    img,label=data
    print(img.shape)
    net=Net()
    output=net(img)
    print(output.shape)
    writer.add_image('test1',img,index)
    reshape_img=torch.reshape(output,(-1,3,30,30))
    writer.add_image('test2',reshape_img,index)
    index+=1