import argparse, time, datetime, sys, os
from copy import deepcopy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.cGAN import *
from utils.datasets import *
from utils.util import *
import hrnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=5, help="size of the batch")
parser.add_argument("--task_num", type=int, default=40, help="number of the Tasks")
parser.add_argument("--meta_batch_size", type=int, default=5, help="size of the meta-learning mini-batch")
parser.add_argument("--outer_step_size", type=float, default=0.1)
parser.add_argument("--n_epochs", type=int, default=41, help="number of epochs of training")
parser.add_argument("--n_FT_epochs", type=int, default=201, help="number of finetune epochs of training")
parser.add_argument("--img_size", type=int, default=512, help="resize the image size")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
parser.add_argument("--train_data_path", type=str, default="", help="path of training data")
parser.add_argument("--FT_data_path", type=str, default="", help="path of fine-tuning data")
parser.add_argument("--model_save_path", type=str, default="", help="path of saved model")

opt = parser.parse_args()

# cuda
cuda = True if torch.cuda.is_available() else False

# Network
generator = hrnet.__dict__["hrnetv2"]()
discriminator = Discriminator()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_size // 2 ** 4, opt.img_size // 2 ** 4)

# loss
criterion_MSE = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()
criterion_SSI = SSI_Loss()
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_L1.cuda()
    criterion_MSE.cuda()
    criterion_SSI.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_OA = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# data transforms
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
]

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------
def train():
    prev_time = time.time()
    # --------------
    # Meta-Training
    # --------------
    for epoch in range(opt.n_epochs):
        train_dataloader = DataLoader(
            ImageDataset_Train(opt.train_data_path, transforms_=transforms_),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)

        # Save Initial Parameters
        init_G_w = deepcopy(generator.state_dict())
        init_D_w = deepcopy(discriminator.state_dict())

        for i, batch in enumerate(train_dataloader):
            meta_G_w = deepcopy(generator.state_dict())
            meta_D_w = deepcopy(discriminator.state_dict())
            
            label = Variable(batch["label"].type(Tensor))

            for j in range(opt.task_num):
                # Task-level Curriculum Learning
                # Introduce Rayleigh noise (sigma ∈ [0.2,0.9])
                sigma = 0.2 + (0.9 - 0.2) * (epoch / opt.n_epochs) * np.random.random()
                # Add Rayleigh noise (rayleigh_noise1) first to fill the pure black background
                rayleigh_noise1 = torch.from_numpy(np.random.rayleigh(scale=0.1, size=(opt.batch_size, opt.img_size, opt.img_size))).float()
                rayleigh_noise2 = torch.from_numpy(np.random.rayleigh(scale=sigma, size=(opt.batch_size, opt.img_size, opt.img_size))).float()
                noise_img = (torch_norm(batch["label"]) + rayleigh_noise1) * rayleigh_noise2
                noise_img = torch.clamp(noise_img, -1.0, 1.0)
                data = Variable((noise_img).type(Tensor))

                # Train Generators
                fake = generator(data)                
                loss_MSE = criterion_MSE(fake, label)
                loss_L1 = criterion_L1(fake, label)
                loss_SSI = criterion_SSI(data, fake)
                loss_G = loss_MSE + 100 * loss_L1 + 10 * loss_SSI

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_OA.zero_grad()
                valid_patch = Variable(Tensor(np.ones((data.size(0), *patch))), requires_grad=False)
                fake_patch = Variable(Tensor(np.zeros((data.size(0), *patch))), requires_grad=False)
                # Real loss
                pred_real = discriminator(label, data)
                loss_real = criterion_MSE(pred_real, valid_patch)
                # Fake loss
                pred_fake = discriminator(fake.detach(), data)
                loss_fake = criterion_MSE(pred_fake, fake_patch)
                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()
                optimizer_OA.step()

                # Log Progress
                batches_done = epoch * len(train_dataloader) * opt.task_num + i * opt.task_num + j
                batches_left = opt.n_epochs * len(train_dataloader) * opt.task_num - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Task %d/%d] [Loss G: %f, MSE: %f, L1: %f, SSI: %f] [Loss OA: %f] ETA: %s"
                    % (epoch, opt.n_epochs, i, len(train_dataloader), j, opt.task_num, 
                    loss_G.item(), loss_MSE.item(), loss_L1.item(), loss_SSI.item(), loss_D.item(), time_left))
            
            # Large-Scale Reptile Update Parameters
            # Meta-level Curriculum Learning
            meta_lr = opt.outer_step_size * (1 - epoch/opt.n_epochs)
            curr_G_w = generator.state_dict()
            curr_D_w = discriminator.state_dict()

            generator.load_state_dict({name: (init_G_w[name] + meta_lr * 
                                    (curr_G_w[name] - meta_G_w[name])) for name in curr_G_w})
            discriminator.load_state_dict({name: (init_D_w[name] + meta_lr * 
                                    (curr_D_w[name] - meta_D_w[name])) for name in curr_D_w})

            init_G_w = deepcopy(generator.state_dict())
            init_D_w = deepcopy(discriminator.state_dict())

            generator.load_state_dict({name: (curr_G_w[name] + meta_lr * 
                                    (curr_G_w[name] - meta_G_w[name])) for name in curr_G_w})
            discriminator.load_state_dict({name: (curr_D_w[name] + meta_lr * 
                                    (curr_D_w[name] - meta_D_w[name])) for name in curr_D_w})
        
        generator.load_state_dict(init_G_w)
        discriminator.load_state_dict(init_D_w)

    # save model after meta-training
    torch.save(generator.state_dict(), "./saved_models/G_Meta_Train_Done.pth")

    # load model of meta-training
    # generator.load_state_dict(torch.load(opt.model_save_path))

    # ---------
    # Finetune
    # ---------
    print("\n=== Meta-Training is over, starting finetune! ===")
    for epoch in range(opt.n_FT_epochs):
        train_dataloader = DataLoader(
            ImageDataset_Train(opt.FT_data_path, transforms_=transforms_),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        for i, batch in enumerate(train_dataloader):
            data = Variable(batch["image"].type(Tensor))
            label = Variable(batch["label"].type(Tensor))

            # Train Generators
            fake = generator(data)                
            loss_MSE = criterion_MSE(fake, label)
            loss_L1 = criterion_L1(fake, label)
            loss_SSI = criterion_SSI(data, fake)
            loss_G = loss_MSE + 100 * loss_L1 + 10 * loss_SSI

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_OA.zero_grad()
            valid_patch = Variable(Tensor(np.ones((data.size(0), *patch))), requires_grad=False)
            fake_patch = Variable(Tensor(np.zeros((data.size(0), *patch))), requires_grad=False)
            # Real loss
            pred_real = discriminator(label, data)
            loss_real = criterion_MSE(pred_real, valid_patch)
            # Fake loss
            pred_fake = discriminator(fake.detach(), data)
            loss_fake = criterion_MSE(pred_fake, fake_patch)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_OA.step()

            # Log Progress
            batches_done = epoch * len(train_dataloader) + i
            batches_left = opt.n_FT_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss G: %f, MSE: %f, L1: %f, SSI: %f] [Loss OA: %f] ETA: %s"
                % (epoch, opt.n_FT_epochs, i, len(train_dataloader),
                loss_G.item(), loss_MSE.item(), loss_L1.item(), loss_SSI.item(), loss_D.item(), time_left))

        # ------------------------------------
        # Write your own validation code here
        # ------------------------------------
        # YourValidCode()

        # save model
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(generator.state_dict(), "./saved_models/generator_%d.pth" % (epoch))


if __name__ == "__main__":
    train()
