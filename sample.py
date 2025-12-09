import torch
from torchvision.utils import save_image
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Sampling from a trained MNISTDiffusion model")
    parser.add_argument('--ckpt', type=str, required=True, help='define checkpoint path')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--ddim_steps', type=int, help='sampling steps of DDIM', default=50)
    parser.add_argument('--cfg_scale', type=float, default=2.0, help='classifier free guidance scale')
    parser.add_argument('--target_label', type=int, default=0, help='target label for sampling')
    parser.add_argument('--sampler', type=str, help='sampler type', default='ddim', choices=['ddpm', 'ddim'])
    parser.add_argument('--no_clip', action='store_true', help='set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu', action='store_true', help='use cpu for sampling')
    parser.add_argument('--output_file', type=str, help='output file name', default='sampled_images.png')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'transformer'], help='Model architecture: unet or transformer')
    args = parser.parse_args()
    return args

def main(args):
    device = "cpu" if args.cpu else "cuda"
    
    # Initialize the model structure
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        ddim_timesteps=args.ddim_steps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
        model_type=args.model_type
    ).to(device)

    # Use ExponentialMovingAverage for the model
    # The training script uses EMA for sampling, so we do the same here for consistency
    model_ema = ExponentialMovingAverage(model, device=device, decay=0.995) # decay value doesn't matter for inference

    # Load the checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # It's common to save the EMA model state for better inference performance
    if "model_ema" in ckpt:
        print("Loading EMA model state.")
        model_ema.load_state_dict(ckpt["model_ema"])
    elif "model" in ckpt:
        print("Loading standard model state.")
        model.load_state_dict(ckpt["model"])
        # If only the base model is available, sync EMA with it
        model_ema = ExponentialMovingAverage(model, device=device, decay=0.995)
    else:
        raise ValueError("Checkpoint does not contain 'model' or 'model_ema' state_dict.")

    model_ema.eval()

    print(f"Sampling {args.n_samples} images using {args.sampler} sampler...")
    if args.sampler == 'ddpm':
        samples = model_ema.module.sampling(
            args.n_samples, 
            clipped_reverse_diffusion=not args.no_clip, 
            cfg_scale=args.cfg_scale, 
            target_label=args.target_label,
            device=device
        )
    elif args.sampler == 'ddim':
        samples = model_ema.module.ddim_sampling(
            args.n_samples, 
            cfg_scale=args.cfg_scale, 
            target_label=args.target_label,
            device=device
        )

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", args.output_file)

    save_image(samples, output_path, nrow=int(math.sqrt(args.n_samples)))
    print(f"Saved samples to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
