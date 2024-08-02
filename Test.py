import cv2
import hrnet
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils.util import *
from utils.cGAN import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device_ids = [0]

# cuda
cuda = True if torch.cuda.is_available() else False

# Network
generator = hrnet.__dict__["hrnetv2"]()

# loss
if cuda:
    generator = generator.cuda()

# data transforms
transforms_ = [
    transforms.Resize((512, 512), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
]

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def Test_images(test_path, save_path):
    # generator.eval()
    transform = transforms.Compose(transforms_)
    image_path = str(test_path)
    image_all = Get_path([".png", ".jpg", '.bmp', '.tif'], image_path)
    # print(image_all)

    for _, img in enumerate(image_all):
        save_img_path = save_path + img.split("/")[-1]
        print(save_img_path)
        
        img = Image.open(img).convert("L")
        img = (transform(img)).unsqueeze(0)
        data = Variable(img.type(Tensor))
        fake = generator(data)
        fake = fake.data.squeeze().cpu().numpy()
        
        res_image = normalization(fake)*255
        cv2.imwrite(save_img_path, res_image)


# ----------
#  Testing
# ----------
def test(model_path, test_path, save_path):
    # load model
    generator.load_state_dict(torch.load(model_path))
    print("\n=== Start testing! ===")
    Test_images(test_path, save_path)


if __name__ == "__main__":
    test(model_path = "",
         test_path  = "",
         save_path  = "")
