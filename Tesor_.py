from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


writer = SummaryWriter('logs')

img=Image.open('val/ants/8124241_36b290d372.jpg')
img_array = np.array(img)
print(img_array.shape)
writer.add_image("test", img_array, 2,dataformats='HWC')
# for i in range(100):
#     writer.add_scalar("y=3x",3*i,i)

writer.close()
