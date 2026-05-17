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
    {"name": "Cinematic",   "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"},
    {"name": "3D Model",    "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"},
    {"name": "Anime",       "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed"},
    {"name": "Digital Art", "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"},
    {"name": "Photographic","prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"},
    {"name": "Pixel art",   "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics"},
    {"name": "Fantasy art", "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy"},
    {"name": "Neonpunk",    "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional"},
    {"name": "Manga",       "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style"},
    {"name": "No Style",    "prompt": "white background {prompt} "},
]

styles = {k["name"]: k["prompt"] for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Anime"
MAX_SEED = np.iinfo(np.int32).max

R_MU = 0.1
R_SIGMA = 0.3


def pil_image_to_data_uri(img, format="PNG"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def sample_r():
    """从正态分布采样 sketch guidance 值，限制在 (0.01, 0.99)。"""
    return float(np.clip(np.random.normal(R_MU, R_SIGMA), 0.01, 0.99))


def generate_image(image_t, prompt, seed, val_r):
    with torch.no_grad():
        c_t = image_t.unsqueeze(0).cuda().float()
        torch.manual_seed(seed)
        _, _, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
        output = model(c_t, prompt, deterministic=False, r=val_r, noise_map=noise)
    return F.to_pil_image(output[0].cpu() * 0.5 + 0.5)


def init_para(selected_images, seed_list_selected, r_list_selected, count):
    return [], [], [], 0


def run(image, prompt, prompt_template, style_name, seed, val_r):
    if image is None:
        ones = Image.new("L", (512, 512), 255)
        uri = pil_image_to_data_uri(ones)
        return ones, gr.update(link=uri), gr.update(link=uri)

    prompt = prompt_template.replace("{prompt}", prompt)
    image = image.convert("RGB")
    image_t = F.to_tensor(image) > 0.5

    seeds, val_rs, images = [], [], []
    for _ in range(16):
        s = random.randint(0, MAX_SEED)
        r = sample_r()
        seeds.append(s)
        val_rs.append(r)
        images.append(generate_image(image_t, prompt, s, r))

    sketch_uri = pil_image_to_data_uri(Image.fromarray(255 - np.array(image)))
    output_uri = pil_image_to_data_uri(images[-1])
    return images, images, seeds, val_rs, gr.update(link=sketch_uri), gr.update(link=output_uri)


def mix(image, image_s, prompt, prompt_template, style_name, seeds_s, rs_s):
    if image is None:
        ones = Image.new("L", (512, 512), 255)
        uri = pil_image_to_data_uri(ones)
        return ones, gr.update(link=uri), gr.update(link=uri)

    prompt = prompt_template.replace("{prompt}", prompt)
    image = image.convert("RGB")
    image_t = F.to_tensor(image) > 0.5

    # 以选中图像的参数为基础，填充到16个槽位
    n = len(seeds_s)
    seeds = [int(s) for s in seeds_s]
    val_rs = list(rs_s)
    avg_r = sum(rs_s) / len(rs_s)

    for i in range(16 - n):
        if i < 4:       # 变异：随机r，从选中seed中随机选
            val_rs.append(sample_r())
            seeds.append(random.choice(seeds_s))
        elif i < 8:     # 平均r，完全随机seed
            val_rs.append(avg_r)
            seeds.append(random.randint(0, MAX_SEED))
        else:           # 完全随机
            val_rs.append(sample_r())
            seeds.append(random.randint(0, MAX_SEED))

    # 打乱顺序后生成
    indices = list(range(16))
    random.shuffle(indices)
    seeds = [seeds[i] for i in indices]
    val_rs = [val_rs[i] for i in indices]

    images = [generate_image(image_t, prompt, seeds[i], val_rs[i]) for i in range(16)]

    sketch_uri = pil_image_to_data_uri(Image.fromarray(255 - np.array(image)))
    output_uri = pil_image_to_data_uri(images[-1])
    # 返回时清空 selection（[], []）
    return (images, images, [], [], seeds, val_rs, [], [],
            gr.update(link=sketch_uri), gr.update(link=output_uri))


def update_canvas(use_line, use_eraser):
    if use_eraser:
        return gr.update(brush_radius=20, brush_color="#ffffff", interactive=True)
    if use_line:
        return gr.update(brush_radius=4, brush_color="#000000", interactive=True)


def show_selected_image(selected_images, imgs, seeds, seeds_s, rs, rs_s, evt: gr.SelectData):
    idx = evt.index
    selected_images.append(imgs[idx])
    seeds_s.append(int(seeds[idx]))
    rs_s.append(rs[idx])
    return selected_images, seeds_s, rs_s


def unselect(selected_images, seeds, seeds_s, rs, rs_s, evt: gr.SelectData):
    idx = evt.index
    selected_images.pop(idx)
    seeds_s.pop(idx)
    rs_s.pop(idx)
    return selected_images, seeds_s, rs_s


def count_clicks(count):
    count += 1
    return count, count


scripts = """
async () => {
    globalThis.theSketchDownloadFunction = () => {
        var link = document.createElement("a");
        link.setAttribute("href", document.getElementById('download_sketch').href);
        link.setAttribute("download", "sketch.png");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        theOutputDownloadFunction();
        return false;
    }

    globalThis.theOutputDownloadFunction = () => {
        var link = document.createElement("a");
        link.setAttribute("href", document.getElementById('download_output').href);
        link.setAttribute("download", "output.png");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        return false;
    }

    globalThis.UNDO_SKETCH_FUNCTION = () => {
        document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(1)').click();
    }

    globalThis.DELETE_SKETCH_FUNCTION = () => {
        document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(2)').click();
    }

    globalThis.togglePencil = () => {
        var el = document.getElementById('my-toggle-pencil');
        el.classList.toggle('clicked');
        document.querySelector("#cb-line > label > input").click();
        if (el.classList.contains('clicked')) {
            document.getElementById('my-toggle-eraser').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        } else {
            document.getElementById('my-toggle-eraser').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        }
    }

    globalThis.toggleEraser = () => {
        var el = document.getElementById('my-toggle-eraser');
        el.classList.toggle('clicked');
        document.querySelector("#cb-eraser > label > input").click();
        if (el.classList.contains('clicked')) {
            document.getElementById('my-toggle-pencil').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        } else {
            document.getElementById('my-toggle-pencil').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        }
    }
}
"""

with gr.Blocks(css="style.css") as demo:
    click_count = gr.State()
    imgs = gr.State()
    selected_images = gr.State()
    seed_list = gr.State()
    r_list = gr.State()
    seed_list_selected = gr.State()
    r_list_selected = gr.State()

    line = gr.Checkbox(label="line", value=False, elem_id="cb-line")
    eraser = gr.Checkbox(label="eraser", value=False, elem_id="cb-eraser")

    with gr.Row(elem_id="main_row"):
        with gr.Column(elem_id="column_input"):
            gr.Markdown("## INPUT", elem_id="input_header")
            sketch_image = gr.Image(
                source="canvas", tool="color-sketch", type="pil",
                image_mode="L", invert_colors=True, shape=(512, 512),
                brush_radius=4, height=440, width=440, brush_color="#000000",
                interactive=True, show_download_button=True,
                elem_id="input_image", show_label=False,
            )
            download_sketch = gr.Button("Download sketch", scale=1, elem_id="download_sketch", visible=False)
            gr.HTML("""
            <div class="button-row">
                <div id="my-div-pencil" class="pad2"><button id="my-toggle-pencil" onclick="return togglePencil(this)"></button></div>
                <div id="my-div-eraser" class="pad2"><button id="my-toggle-eraser" onclick="return toggleEraser(this)"></button></div>
                <div class="pad2"><button id="my-button-undo" onclick="return UNDO_SKETCH_FUNCTION(this)"></button></div>
                <div class="pad2"><button id="my-button-clear" onclick="return DELETE_SKETCH_FUNCTION(this)"></button></div>
                <div class="pad2"><button href="TODO" download="image" id="my-button-down" onclick='return theSketchDownloadFunction()'></button></div>
            </div>
            """)
            prompt = gr.Textbox(label="Prompt", value="", show_label=True)
            with gr.Row():
                style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME, scale=1)
                prompt_temp = gr.Textbox(label="Prompt Style Template", value=styles[DEFAULT_STYLE_NAME], scale=2, max_lines=1, visible=False)
            with gr.Row():
                val_r = gr.Slider(label="Sketch guidance:", minimum=0, maximum=1, value=0.4, step=0.01, scale=3, visible=False)
                seed = gr.Textbox(label="Seed", value=42, scale=1, min_width=50, visible=False)
                run_button = gr.Button("Run", min_width=50)

        with gr.Column(elem_id="column_output"):
            gr.Markdown("## OUTPUT", elem_id="output_header")
            images_gallery = gr.Gallery(allow_preview=False, show_label=False, columns=4, row=4)
            download_output = gr.Button("Download output", elem_id="download_output")
            gr.Markdown("## Selected Image", elem_id="selected_header")
            selected_gallery = gr.Gallery(allow_preview=False, show_label=False)
            final_Image = gr.Gallery(visible=False)
            mix_button = gr.Button("Mix")
            chose_button = gr.Button("Chose")
            count = gr.Number(label="Mix回数")

    demo.load(None, None, None, _js=scripts)
    demo.load(init_para,
              [selected_images, seed_list_selected, r_list_selected, click_count],
              [selected_images, seed_list_selected, r_list_selected, click_count])

    eraser.change(fn=lambda x: gr.update(value=not x), inputs=[eraser], outputs=[line], queue=False, api_name=False
                  ).then(update_canvas, [line, eraser], [sketch_image])
    line.change(fn=lambda x: gr.update(value=not x), inputs=[line], outputs=[eraser], queue=False, api_name=False
                ).then(update_canvas, [line, eraser], [sketch_image])

    style.change(lambda x: styles[x], inputs=[style], outputs=[prompt_temp], queue=False, api_name=False)

    sketch_inputs = [sketch_image, prompt, prompt_temp, style, seed, val_r]
    image_outputs = [images_gallery, imgs, seed_list, r_list, download_sketch, download_output]
    prompt.submit(fn=run, inputs=sketch_inputs, outputs=image_outputs, api_name=False)
    run_button.click(run, inputs=sketch_inputs, outputs=image_outputs, api_name=False)

    select_inputs = [selected_images, imgs, seed_list, seed_list_selected, r_list, r_list_selected]
    select_outputs = [selected_gallery, seed_list_selected, r_list_selected]
    images_gallery.select(show_selected_image, select_inputs, select_outputs)
    selected_gallery.select(unselect, [selected_images, seed_list, seed_list_selected, r_list, r_list_selected], select_outputs)

    rerun_inputs = [sketch_image, selected_images, prompt, prompt_temp, style, seed_list_selected, r_list_selected]
    rerun_outputs = [images_gallery, imgs, selected_images, selected_images, seed_list, r_list,
                     seed_list_selected, r_list_selected, download_sketch, download_output]
    mix_button.click(count_clicks, inputs=[click_count], outputs=[count, click_count])
    mix_button.click(mix, inputs=rerun_inputs, outputs=rerun_outputs, queue=False, api_name=False)
    mix_button.click(lambda: ([], [], []), outputs=[seed_list_selected, r_list_selected, selected_gallery])

    chose_button.click(
        lambda imgs: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), imgs),
        inputs=[selected_images],
        outputs=[images_gallery, selected_gallery, final_Image, final_Image]
    )

if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True)
