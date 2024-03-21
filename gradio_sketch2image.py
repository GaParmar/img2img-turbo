import random
import numpy as np
from PIL import Image
import base64
from io import BytesIO

import torch
import torchvision.transforms.functional as F
import gradio as gr

from src.pix2pix_turbo import Pix2Pix_Turbo

model = Pix2Pix_Turbo("sketch_to_image_stochastic")

style_list = [
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
    },
]

styles = {k["name"]: k["prompt"] for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Fantasy art"
MAX_SEED = np.iinfo(np.int32).max


def pil_image_to_data_uri(img, format="PNG"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def run(image, prompt, prompt_template, style_name, seed, val_r):
    print(f"prompt: {prompt}")
    print("sketch updated")
    if image is None:
        ones = Image.new("L", (512, 512), 255)
        temp_uri = pil_image_to_data_uri(ones)
        return ones, gr.update(link=temp_uri), gr.update(link=temp_uri)
    prompt = prompt_template.replace("{prompt}", prompt)
    image = image.convert("RGB")
    image_t = F.to_tensor(image) > 0.5
    print(f"r_val={val_r}, seed={seed}")
    with torch.no_grad():
        c_t = image_t.unsqueeze(0).cuda().float()
        torch.manual_seed(seed)
        B, C, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
        output_image = model(c_t, prompt, deterministic=False, r=val_r, noise_map=noise)
    output_pil = F.to_pil_image(output_image[0].cpu() * 0.5 + 0.5)
    input_sketch_uri = pil_image_to_data_uri(Image.fromarray(255 - np.array(image)))
    output_image_uri = pil_image_to_data_uri(output_pil)
    return (
        output_pil,
        gr.update(link=input_sketch_uri),
        gr.update(link=output_image_uri),
    )


def update_canvas(use_line, use_eraser):
    if use_eraser:
        _color = "#ffffff"
        brush_size = 20
    if use_line:
        _color = "#000000"
        brush_size = 4
    return gr.update(brush_radius=brush_size, brush_color=_color, interactive=True)


def upload_sketch(file):
    _img = Image.open(file.name)
    _img = _img.convert("L")
    return gr.update(value=_img, source="upload", interactive=True)


scripts = """
async () => {
    globalThis.theSketchDownloadFunction = () => {
        console.log("test")
        var link = document.createElement("a");
        dataUri = document.getElementById('download_sketch').href
        link.setAttribute("href", dataUri)
        link.setAttribute("download", "sketch.png")
        document.body.appendChild(link); // Required for Firefox
        link.click();
        document.body.removeChild(link); // Clean up

        // also call the output download function
        theOutputDownloadFunction();
      return false
    }

    globalThis.theOutputDownloadFunction = () => {
        console.log("test output download function")
        var link = document.createElement("a");
        dataUri = document.getElementById('download_output').href
        link.setAttribute("href", dataUri);
        link.setAttribute("download", "output.png");
        document.body.appendChild(link); // Required for Firefox
        link.click();
        document.body.removeChild(link); // Clean up
      return false
    }

    globalThis.UNDO_SKETCH_FUNCTION = () => {
        console.log("undo sketch function")
        var button_undo = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(1)');
        // Create a new 'click' event
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        button_undo.dispatchEvent(event);
    }

    globalThis.DELETE_SKETCH_FUNCTION = () => {
        console.log("delete sketch function")
        var button_del = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(2)');
        // Create a new 'click' event
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        button_del.dispatchEvent(event);
    }

    globalThis.togglePencil = () => {
        el_pencil = document.getElementById('my-toggle-pencil');
        el_pencil.classList.toggle('clicked');
        // simulate a click on the gradio button
        btn_gradio = document.querySelector("#cb-line > label > input");
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        btn_gradio.dispatchEvent(event);
        if (el_pencil.classList.contains('clicked')) {
            document.getElementById('my-toggle-eraser').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        }
        else {
            document.getElementById('my-toggle-eraser').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        }
    }

    globalThis.toggleEraser = () => {
        element = document.getElementById('my-toggle-eraser');
        element.classList.toggle('clicked');
        // simulate a click on the gradio button
        btn_gradio = document.querySelector("#cb-eraser > label > input");
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        btn_gradio.dispatchEvent(event);
        if (element.classList.contains('clicked')) {
            document.getElementById('my-toggle-pencil').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        }
        else {
            document.getElementById('my-toggle-pencil').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        }
    }
}
"""

with gr.Blocks(css="style.css") as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h2><a href="https://github.com/GaParmar/img2img-turbo">One-Step Image Translation with Text-to-Image Models</a></h2>
                <div>
                    <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                        <a href='https://gauravparmar.com/'>Gaurav Parmar, </a>
                        &nbsp;
                        <a href='https://taesung.me/'> Taesung Park,</a>
                        &nbsp;
                        <a href='https://www.cs.cmu.edu/~srinivas/'>Srinivasa Narasimhan, </a>
                        &nbsp;
                        <a href='https://www.cs.cmu.edu/~junyanz/'> Jun-Yan Zhu </a>
                    </div>
                </div>
                </br>
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <a href='https://arxiv.org/abs/2403.12036'>
                        <img src="https://img.shields.io/badge/arXiv-2403.12036-red">
                    </a>
                    &nbsp;
                    <a href='https://github.com/GaParmar/img2img-turbo'>
                        <img src='https://img.shields.io/badge/github-%23121011.svg'>
                    </a>
                    &nbsp;
                    <a href='https://github.com/GaParmar/img2img-turbo/blob/main/LICENSE'>
                        <img src='https://img.shields.io/badge/license-MIT-lightgrey'>
                    </a>
                </div>
            </div>
        </div>
        <div>
        </br>
        </div>
        """
    )

    # these are hidden buttons that are used to trigger the canvas changes
    line = gr.Checkbox(label="line", value=False, elem_id="cb-line")
    eraser = gr.Checkbox(label="eraser", value=False, elem_id="cb-eraser")
    with gr.Row(elem_id="main_row"):
        with gr.Column(elem_id="column_input"):
            gr.Markdown("## INPUT", elem_id="input_header")
            image = gr.Image(
                source="canvas",
                tool="color-sketch",
                type="pil",
                image_mode="L",
                invert_colors=True,
                shape=(512, 512),
                brush_radius=4,
                height=440,
                width=440,
                brush_color="#000000",
                interactive=True,
                show_download_button=True,
                elem_id="input_image",
                show_label=False,
            )
            download_sketch = gr.Button(
                "Download sketch", scale=1, elem_id="download_sketch"
            )

            gr.HTML(
                """
            <div class="button-row">
                <div id="my-div-pencil" class="pad2"> <button id="my-toggle-pencil" onclick="return togglePencil(this)"></button> </div>
                <div id="my-div-eraser" class="pad2"> <button id="my-toggle-eraser" onclick="return toggleEraser(this)"></button> </div>
                <div class="pad2"> <button id="my-button-undo" onclick="return UNDO_SKETCH_FUNCTION(this)"></button> </div>
                <div class="pad2"> <button id="my-button-clear" onclick="return DELETE_SKETCH_FUNCTION(this)"></button> </div>
                <div class="pad2"> <button href="TODO" download="image" id="my-button-down" onclick='return theSketchDownloadFunction()'></button> </div>
            </div>
            """
            )
            # gr.Markdown("## Prompt", elem_id="tools_header")
            prompt = gr.Textbox(label="Prompt", value="", show_label=True)
            with gr.Row():
                style = gr.Dropdown(
                    label="Style",
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                    scale=1,
                )
                prompt_temp = gr.Textbox(
                    label="Prompt Style Template",
                    value=styles[DEFAULT_STYLE_NAME],
                    scale=2,
                    max_lines=1,
                )

            with gr.Row():
                val_r = gr.Slider(
                    label="Sketch guidance: ",
                    show_label=True,
                    minimum=0,
                    maximum=1,
                    value=0.4,
                    step=0.01,
                    scale=3,
                )
                seed = gr.Textbox(label="Seed", value=42, scale=1, min_width=50)
                randomize_seed = gr.Button("Random", scale=1, min_width=50)

        with gr.Column(elem_id="column_process", min_width=50, scale=0.4):
            gr.Markdown("## pix2pix-turbo", elem_id="description")
            run_button = gr.Button("Run", min_width=50)

        with gr.Column(elem_id="column_output"):
            gr.Markdown("## OUTPUT", elem_id="output_header")
            result = gr.Image(
                label="Result",
                height=440,
                width=440,
                elem_id="output_image",
                show_label=False,
                show_download_button=True,
            )
            download_output = gr.Button("Download output", elem_id="download_output")
            gr.Markdown("### Instructions")
            gr.Markdown("**1**. Enter a text prompt (e.g. cat)")
            gr.Markdown("**2**. Start sketching")
            gr.Markdown("**3**. Change the image style using a style template")
            gr.Markdown("**4**. Adjust the effect of sketch guidance using the slider")
            gr.Markdown("**5**. Try different seeds to generate different results")

    eraser.change(
        fn=lambda x: gr.update(value=not x),
        inputs=[eraser],
        outputs=[line],
        queue=False,
        api_name=False,
    ).then(update_canvas, [line, eraser], [image])
    line.change(
        fn=lambda x: gr.update(value=not x),
        inputs=[line],
        outputs=[eraser],
        queue=False,
        api_name=False,
    ).then(update_canvas, [line, eraser], [image])

    demo.load(None, None, None, _js=scripts)
    randomize_seed.click(
        lambda x: random.randint(0, MAX_SEED),
        inputs=[],
        outputs=seed,
        queue=False,
        api_name=False,
    )
    inputs = [image, prompt, prompt_temp, style, seed, val_r]
    outputs = [result, download_sketch, download_output]
    prompt.submit(fn=run, inputs=inputs, outputs=outputs, api_name=False)
    style.change(
        lambda x: styles[x],
        inputs=[style],
        outputs=[prompt_temp],
        queue=False,
        api_name=False,
    ).then(
        fn=run,
        inputs=inputs,
        outputs=outputs,
        api_name=False,
    )
    val_r.change(run, inputs=inputs, outputs=outputs, queue=False, api_name=False)
    run_button.click(fn=run, inputs=inputs, outputs=outputs, api_name=False)
    image.change(run, inputs=inputs, outputs=outputs, queue=False, api_name=False)

if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True)
