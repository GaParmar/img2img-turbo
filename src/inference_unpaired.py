import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--model_name', type=str, default='day_to_night', help='name of the model to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    args = parser.parse_args()

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name)
    model.unet.enable_xformers_memory_efficient_attention()

    if args.image_prep == "resize_512x512":
        T_val = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    input_image = Image.open(args.input_image).convert('RGB')
    # translate the image
    with torch.no_grad():
        x_t = T_val(input_image).unsqueeze(0).cuda()
        output = model(x_t)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

    # save the output image
    bname = os.path.basename(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    output_pil.save(os.path.join(args.output_dir, bname))
