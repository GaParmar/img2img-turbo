import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from image_prep import canny_from_pil

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='edge_to_image', help='name of the model to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    # initialize the model
    model = Pix2Pix_Turbo(args.model_name)

    # make sure that the input image is a multiple of 8
    input_image = Image.open(args.input_image).convert('RGB')
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    # translate the image
    with torch.no_grad():
        if args.model_name == 'edge_to_image':
            canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
            c_t = transforms.ToTensor()(canny).unsqueeze(0).cuda()
            output_image = model(c_t, args.prompt)
        
        if args.model_name == 'sketch_to_image_stochastic':
            image_t = F.to_tensor(input_image) < 0.5
            c_t = image_t.unsqueeze(0).cuda().float()
            torch.manual_seed(args.seed)
            B,C,H,W = c_t.shape
            noise = torch.randn((1,4,H//8, W//8), device=c_t.device)
            output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)
        
        output_pil = transforms.ToPILImage()(output_image[0].cpu()*0.5+0.5)
    
    # save the output image
    bname = os.path.basename(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    output_pil.save(os.path.join(args.output_dir, bname))
