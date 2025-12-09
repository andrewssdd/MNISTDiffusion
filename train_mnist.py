import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=2e-4)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples_per_target',type = int,help = 'define sampling amounts after every epoch trained per target',default=5)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of model',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--sampler',type = str,help = 'sampler type',default='ddim',choices=['ddpm', 'ddim'])
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--uncond_prob',type=float,default=0.1)
    parser.add_argument('--cfg_scale',type=float,default=2.0)
    parser.add_argument('--n_classes',type=int,default=11)
    parser.add_argument('--model_type',type=str,default='unet',choices=['unet', 'transformer'],help='Model architecture: unet or transformer')
    parser.add_argument('--trainer',type=str,default='ddpm',choices=['ddpm', 'rectified_flow'],help='Training method for diffusion model')

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda"
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4],
                n_classes=args.n_classes,
                uncond_prob=args.uncond_prob,
                model_type=args.model_type,
                diffusion_type=args.trainer).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    warmup_steps = 1000
    lambda_fn = lambda step: min(step / warmup_steps, 1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
    loss_fn=nn.MSELoss(reduction='mean')

    # load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    # targets
    train_targets = []
    for image,target in train_dataloader:
        train_targets += target.tolist()
    targets = []
    for t in train_targets:
        if t not in targets:
            targets.append(t)
    targets.sort()

    # training loop
    global_steps=0
    for i in range(args.epochs):
        model.train()
        for j,(image,target) in enumerate(train_dataloader):
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            target=target.to(device)
            pred=model(image,noise,target)
            
            if args.trainer == 'rectified_flow':
                loss=loss_fn(pred, noise - image)
            else:
                loss=loss_fn(pred,noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}

        os.makedirs("results",exist_ok=True)
        torch.save(ckpt,"results/steps_{:0>8}.pt".format(global_steps))

        # generate samples
        model_ema.eval()
        labels = [t for t in targets for _ in range(args.n_samples_per_target)]
        if args.trainer == 'rectified_flow':
            samples=model_ema.module.sampling(len(labels), labels,args.cfg_scale,device=device)
        elif args.sampler == 'ddpm':
            samples=model_ema.module.sampling(len(labels), labels,args.cfg_scale,clipped_reverse_diffusion=not args.no_clip,device=device)
        elif args.sampler == 'ddim':
            samples=model_ema.module.ddim_sampling(len(labels), labels,args.cfg_scale,device=device)
        else:
            raise NotImplementedError()
        save_image(samples,"results/{}_steps_{:0>8}.png".format(args.model_type, global_steps),nrow=args.n_samples_per_target)

if __name__=="__main__":
    args=parse_args()
    main(args)