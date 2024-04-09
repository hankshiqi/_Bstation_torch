from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image



img_path = 'val/bees/26589803_5ba7000313.jpg'
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_=transforms.ToTensor()(img)
writer.add_image('Tensorimg',tensor_)
writer.close()

