from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
import os
import requests
from tqdm import tqdm
CACHE_DIR = "./checkpoints"
ckpt_folder="checkpoints"
os.makedirs(ckpt_folder, exist_ok=True)

AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", cache_dir=CACHE_DIR)
try:
    CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder", cache_dir=CACHE_DIR).cuda()
except:
    print("Error Occured")
try:
    AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae", cache_dir=CACHE_DIR)
except:
    print("Error Occured")
try:
    UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet", cache_dir=CACHE_DIR)
except:
    print("Error Occured")

url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
if not os.path.exists(outf):
    print(f"Downloading checkpoint to {outf}")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
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

url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
os.makedirs(ckpt_folder, exist_ok=True)
outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
if not os.path.exists(outf):
    print(f"Downloading checkpoint to {outf}")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
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