import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset_Train(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.image_files = sorted(glob.glob(root + "/image/*.*"))
        self.label_files = sorted(glob.glob(root + "/label/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.image_files[index % len(self.image_files)]).convert("L")
        lab = Image.open(self.label_files[index % len(self.label_files)]).convert("L")

        if (np.random.random() < 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
        
        factor = np.random.randint(-10,10)
        img = img.rotate(factor)
        lab = lab.rotate(factor)

        img = self.transform(img)
        lab = self.transform(lab)

        return {"image": img, "label": lab}

    def __len__(self):
        return len(self.image_files)
