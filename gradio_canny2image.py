import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr
from src.image_prep import canny_from_pil
from src.pix2pix_turbo import Pix2Pix_Turbo

model = Pix2Pix_Turbo("edge_to_image")


def process(input_image, prompt, low_threshold, high_threshold):
    # resize to be a multiple of 8
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height))
    canny = canny_from_pil(input_image, low_threshold, high_threshold)
    with torch.no_grad():
        c_t = transforms.ToTensor()(canny).unsqueeze(0).cuda()
        output_image = model(c_t, prompt)
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    # flippy canny values, map all 0s to 1s and 1s to 0s
    canny_viz = 1 - (np.array(canny) / 255)
    canny_viz = Image.fromarray((canny_viz * 255).astype(np.uint8))
    return canny_viz, output_pil


if __name__ == "__main__":
    # load the model
    with gr.Blocks() as demo:
        gr.Markdown("# Pix2pix-Turbo: **Canny Edge -> Image**")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources="upload", type="pil")
                prompt = gr.Textbox(label="Prompt")
                low_threshold = gr.Slider(
                    label="Canny low threshold",
                    minimum=1,
                    maximum=255,
                    value=100,
                    step=10,
                )
                high_threshold = gr.Slider(
                    label="Canny high threshold",
                    minimum=1,
                    maximum=255,
                    value=200,
                    step=10,
                )
                run_button = gr.Button(value="Run")
            with gr.Column():
                result_canny = gr.Image(type="pil")
            with gr.Column():
                result_output = gr.Image(type="pil")

        prompt.submit(
            fn=process,
            inputs=[input_image, prompt, low_threshold, high_threshold],
            outputs=[result_canny, result_output],
        )
        low_threshold.change(
            fn=process,
            inputs=[input_image, prompt, low_threshold, high_threshold],
            outputs=[result_canny, result_output],
        )
        high_threshold.change(
            fn=process,
            inputs=[input_image, prompt, low_threshold, high_threshold],
            outputs=[result_canny, result_output],
        )
        run_button.click(
            fn=process,
            inputs=[input_image, prompt, low_threshold, high_threshold],
            outputs=[result_canny, result_output],
        )

    demo.queue()
    demo.launch(debug=True, share=False)
