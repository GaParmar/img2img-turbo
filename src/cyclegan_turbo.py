import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a=vae_b2a
    
    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction=="a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample()*_vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a=vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction=="a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor ).sample).clamp(-1,1)
        return x_decoded


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, name, ckpt_folder="checkpoints"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo",subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if name=="day_to_night":
            # download the checkopoint from the url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "day2night.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            
            sd = torch.load(outf)
            lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
            lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
            lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
            unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
            unet.add_adapter(lora_conf_others, adapter_name="default_others")
            for n,p in unet.named_parameters():
                name_sd = n.replace(f".default_encoder.weight", ".weight")
                if "lora" in n and "default_encoder" in n:
                    p.data.copy_(sd["sd_encoder"][name_sd])
            for n,p in unet.named_parameters():
                name_sd = n.replace(f".default_decoder.weight", ".weight")
                if "lora" in n and "default_decoder" in n:
                    p.data.copy_(sd["sd_decoder"][name_sd])
            for n,p in unet.named_parameters():
                name_sd = n.replace(f".default_others.weight", ".weight")
                if "lora" in n and "default_others" in n:
                    p.data.copy_(sd["sd_other"][name_sd])
            unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

            vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
            vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
            vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.ignore_skip = False
            vae_lora_config = LoraConfig(r=4, init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            vae.decoder.gamma = 1
            vae_b2a = copy.deepcopy(vae)
            vae_enc = VAE_encode(vae, vae_b2a=vae_b2a)
            vae_enc.load_state_dict(sd["sd_vae_enc"])
            vae_dec = VAE_decode(vae, vae_b2a=vae_b2a)
            vae_dec.load_state_dict(sd["sd_vae_dec"])
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the night"
        vae_enc.cuda()
        vae_dec.cuda()
        unet.cuda()
        unet.enable_xformers_memory_efficient_attention()
        self.unet, self.vae_enc, self.vae_dec = unet, vae_enc, vae_dec


    def forward(self, x_t, direction):
        assert direction in ["a2b", "b2a"]
        caption_tokens = self.tokenizer(self.caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]
        x_t_enc = self.vae_enc(x_t, direction=direction)
        model_pred = self.unet(x_t_enc, self.timesteps,encoder_hidden_states=caption_enc,).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, x_t_enc, return_dict=True).prev_sample
        output = self.vae_dec(x_denoised, direction=direction)
        return output
