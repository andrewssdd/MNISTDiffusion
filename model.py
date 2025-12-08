import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm

class MNISTDiffusion(nn.Module):
    def __init__(self,image_size,in_channels,n_classes=11,time_embedding_dim=256,timesteps=1000,ddim_timesteps=100,base_dim=32,dim_mults= [1, 2, 4, 8],uncond_prob=0.1):
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size
        self.ddim_timesteps=ddim_timesteps
        self.n_classes=n_classes
        self.uncond_prob=uncond_prob

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults,n_classes)

    def forward(self,x,noise,target):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        
        #randomly replace target with unconditional class token to implement cfg
        target=target if torch.rand(1)>self.uncond_prob else torch.tensor(self.n_classes-1).to(target.device)

        pred_noise=self.model(x_t,t,target)

        return pred_noise

    @torch.no_grad()
    def sampling(self,n_samples,target_label,cfg_scale,clipped_reverse_diffusion=True,device="cuda"):
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        if type(target_label) is int:
            target_label=torch.tensor([target_label for _ in range(n_samples)]).to(device)
        else:
            target_label=torch.tensor(target_label).to(device) # already one label per sample

        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise,target_label,cfg_scale)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise,target_label,cfg_scale)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t

    @torch.no_grad()
    def ddim_sampling(self, n_samples,target_label,cfg_scale, device="cuda"):
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        if type(target_label) is int:
            target_label=torch.tensor([target_label for _ in range(n_samples)]).to(device)
        else:
            target_label=torch.tensor(target_label).to(device)
        
        # Generates times from 0 to T-1 and flips to get [T-1, ..., 0]
        # This ensures endpoints 0 and T-1 are included and spacing is symmetric
        times = torch.linspace(0, self.timesteps - 1, steps=self.ddim_timesteps).long().flip(0).to(device)
        
        # Iterate forward through the descending times array
        for i in tqdm(range(self.ddim_timesteps), desc="DDIM Sampling"):
            time = times[i]
            
            # Look ahead for the next step
            # If we are at the last step, prev_time is -1
            prev_time = times[i + 1] if i < self.ddim_timesteps - 1 else -1
            
            t = torch.full((n_samples,), time, device=device, dtype=torch.long)
            # prev_t can be scalar -1 if i is last, or tensor. 
            # Ideally keep it consistent. 
            if type(prev_time) is torch.Tensor:
                prev_time = prev_time.item()
            
            prev_t = torch.full((n_samples,), prev_time, device=device, dtype=torch.long)

            x_t = self._ddim_reverse_diffusion(x_t, t, prev_t,target_label,cfg_scale)
            
        x_t = (x_t + 1.) / 2.
        return x_t

    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise,target_label,cfg_scale):
        uncond_label=torch.full_like(target_label,self.n_classes-1)
        uncond_pred=self.model(x_t,t,uncond_label)
        cond_pred=self.model(x_t,t,target_label)

        pred=torch.lerp(uncond_pred,cond_pred,cfg_scale)


        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise,target_label,cfg_scale): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        uncond_label=torch.full_like(target_label,self.n_classes-1)
        uncond_pred=self.model(x_t,t,uncond_label)
        cond_pred=self.model(x_t,t,target_label)

        pred=torch.lerp(uncond_pred,cond_pred,cfg_scale)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    
    @torch.no_grad()
    def _ddim_reverse_diffusion(self, x_t, t, prev_t, target_label, cfg_scale):

        # Conditoinal prediction
        uncond_label=torch.full_like(target_label,self.n_classes-1)
        uncond_pred=self.model(x_t,t,uncond_label)
        cond_pred=self.model(x_t,t,target_label)
        
        # CFG Formula: uncond + scale * (cond - uncond)
        pred = torch.lerp(uncond_pred, cond_pred, cfg_scale)

        # 2. Get Alpha values
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        
        alpha_t_cumprod_prev = torch.ones_like(alpha_t_cumprod)
        prev_t_positive_mask = prev_t >= 0
        if prev_t_positive_mask.any():
            valid_prev_t = prev_t[prev_t_positive_mask]
            alpha_t_cumprod_prev[prev_t_positive_mask] = self.alphas_cumprod.gather(-1, valid_prev_t).reshape(-1, 1, 1, 1)

        # 3. Predict x_0 (original)
        # x_0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        beta_prod_t = 1. - alpha_t_cumprod
        x_0_pred = (x_t - beta_prod_t.sqrt() * pred) / alpha_t_cumprod.sqrt()
        
        # 4. Clip x_0 (Dynamic Thresholding)
        x_0_pred.clamp_(-1., 1.)
        
        # Re-derive epsilon from the clamped x_0 for consistency. This ensures the update step remains valid for the clamped data
        pred_rederived = (x_t - alpha_t_cumprod.sqrt() * x_0_pred) / beta_prod_t.sqrt()

        # 5. DDIM Update Step
        # x_{t-1} = sqrt(alpha_{t-1}) * x_0_pred + sqrt(1 - alpha_{t-1}) * pred_rederived
        # Note: We use pred_rederived here instead of 'pred'
        std_dev_term = torch.sqrt(1. - alpha_t_cumprod_prev)
        x_t_minus_1 = torch.sqrt(alpha_t_cumprod_prev) * x_0_pred + std_dev_term * pred_rederived
        
        return x_t_minus_1