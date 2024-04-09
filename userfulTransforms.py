from  PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
#ToTensor
img = Image.open('val/bees/54736755_c057723f64.jpg')
transform_toTensor = transforms.ToTensor()(img)
print(transform_toTensor[0][0][0])
#Normalize归一化
transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transform_toTensor)
print(transform_norm[0][0][0])
img_resize=transforms.Resize((224,224))(img)
img_resize=transforms.ToTensor()(img_resize)
writer.add_image("image",img_resize,0)
img.show(img_resize)

#Compose
