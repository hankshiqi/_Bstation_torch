from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.image_path[index]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.image_path)


root_dir='val'
label_dir='ants'
ants_dataset=MyDataset(root_dir,label_dir)


