# img2img-turbo

### [**Paper**](https://arxiv.org/abs/2403.12036) | [**Sketch2Image Demo**](https://huggingface.co/spaces/gparmar/img2img-turbo-sketch)
#### **Quick start:** [**Running Locally**](#getting-started) | [**Gradio (locally hosted)**](#gradio-demo)

<p align="center" width="800">
<img src="https://raw.githubusercontent.com/GaParmar/img2img-turbo/main/assets/cat_2x.gif" />
</p>
<p align="center" width="800">
<img src="https://raw.githubusercontent.com/GaParmar/img2img-turbo/main/assets/fish_2x.gif" />
</p>




We propose a general method for adapting a single-step diffusion model, such as SD-Turbo, to new tasks and domains through adversarial learning. This enables us to leverage the internal knowledge of pre-trained diffusion models while achieving efficient inference (e.g., for 512x512 image, 0.29 seconds on A6000, and 0.11 seconds on A100). Our single-step image-to-image translation models, called **CycleGAN-Turbo** and **pix2pix-turbo**, can synthesize realistic outputs for unpaired and paired settings. Our one-step model can perform various image-to-image translation tasks for both paired and unpaired settings. CycleGAN-Turbo outperforms existing GAN-based and diffusion-based methods, while pix2pix-Turbo is on par with recent works such as ControlNet for Sketch2Photo and Edge2Image, but with one-step inference. 



<br>
<div>
<p align="center">
<img src='assets/teaser_results.jpg' align="center" width=1000px>
</p>
</div>



## Results

### Paired Translation
**Edge to Image**
<div>
<p align="center">
<img src='assets/edge_to_image_results.jpg' align="center" width=800px>
</p>
</div>

<!-- **Sketch to Image**
TODO -->

### Unpaired Translation on Driving Images

**Day to Night**
<div> <p align="center">
<img src='assets/day2night_results.jpg' align="center" width=800px>
</p> </div>

**Night to Day**
<div><p align="center">
<img src='assets/night2day_results.jpg' align="center" width=800px>
</p> </div>

**Clear to Rainy**
<div>
<p align="center">
<img src='assets/clear2rainy_results.jpg' align="center" width=800px>
</p>
</div>

**Rainy to Clear**
<div>
<p align="center">
<img src='assets/rainy2clear.jpg' align="center" width=800px>
</p>
</div>
<hr>


## Method
**Our Generator Architecture:**
We tightly integrate three separate modules in the original latent diffusion models into a single end-to-end network with small trainable weights. This architecture allows us to translate the input image x to the output y, while retaining the input scene structure. We use LoRA adapters in each module, introduce skip connections and Zero-Convs between input and output, and retrain the first layer of the U-Net. Blue boxes indicate trainable layers. Semi-transparent layers are frozen. The same generator can be used for various GAN objectives.
<div>
<p align="center">
<img src='assets/method.jpg' align="center" width=900px>
</p>
</div>


## Getting Started
**Environment Setup**
- We provide a [conda env file](environment.yml) that contains all the required dependencies.
    ```
    conda env create -f environment.yaml
    ```
- Following this, you can activate the conda environment with the command below. 
  ```
  conda activate img2img-turbo
  ```


**Paired Image Translation (pix2pix-turbo)**
- The following command takes an image file and a prompt as inputs, extracts the canny edges, and saves the results in the directory specified.
    ```bash
    python src/inference_paired.py --model "edge_to_image" \
        --input_image "assets/bird.png" \
        --prompt "a blue bird" \
        --output_dir "outputs"
    ```

- The following command takes a sketch and a prompt as inputs, and saves the results in the directory specified.
    ```bash
    python src/inference_paired.py --model "sketch_to_image_stochastic" \
    --input_image "assets/sketch.png" --gamma 0.4 \
    --prompt "ethereal fantasy concept art of an asteroid. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy" \
    --output_dir "outputs"
    ```

**Unpaired Image Translation (CycleGAN-Turbo)**
- The following command takes an image file as input, and saves the results in the directory specified.
    ```bash
    python src/inference_unpaired.py --model "day_to_night" \
        --input_image "assets/day.png" --output_dir "outputs"
    ```


## Acknowledgment
Our work uses the Stable Diffusion-Turbo as the base model with the following [LICENSE](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE).
