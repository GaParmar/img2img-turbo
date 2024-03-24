import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name, ckpt_folder="checkpoints"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae
        if pretrained_name == "day_to_night":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the night"
            self.direction = "a2b"
        elif pretrained_name == "night_to_day":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        elif pretrained_name == "clear_to_rainy":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in heavy rain"
            self.direction = "a2b"
        elif pretrained_name == "rainy_to_clear":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    def forward(self, x_t):
        caption_tokens = self.tokenizer(self.caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]
        x_t_enc = self.vae_enc(x_t, direction=self.direction)
        model_pred = self.unet(x_t_enc, self.timesteps, encoder_hidden_states=caption_enc,).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, x_t_enc, return_dict=True).prev_sample
        output = self.vae_dec(x_denoised, direction=self.direction)
        return output
