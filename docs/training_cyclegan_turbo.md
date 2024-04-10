## Training with Unpaired Data (CycleGAN-turbo)
Here, we show how to train a CycleGAN-turbo model using unpaired data.
We will use the [horse2zebra dataset](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md) introduced by [CycleGAN](https://junyanz.github.io/CycleGAN/) as an example dataset.


### Step 1. Get the Dataset
- First download the horse2zebra dataset from [here](https://www.cs.cmu.edu/~img2img-turbo/data/my_horse2zebra.zip) using the command below.
    ```
    bash scripts/download_horse2zebra.sh
    ```

- Our training scripts expect the dataset to be in the following format:
    ```
    data
    ├── dataset_name
    │   ├── train_A
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── train_B
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   └── fixed_prompt_a.txt
    |   └── fixed_prompt_b.txt
    |
    |   ├── test_A
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── test_B
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    ```
- The `fixed_prompt_a.txt` and `fixed_prompt_b.txt` files contain the **fixed caption** used for the source and target domains respectively.


### Step 2. Train the Model
- Initialize the `accelerate` environment with the following command:
    ```
    accelerate config
    ```

- Run the following command to train the model. 
    ```
    export NCCL_P2P_DISABLE=1
    accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
        --pretrained_model_name_or_path="stabilityai/sd-turbo" \
        --output_dir="output/cyclegan_turbo/my_horse2zebra" \
        --dataset_folder "data/my_horse2zebra" \
        --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize" \
        --learning_rate="1e-5" --max_train_steps=25000 \
        --train_batch_size=1 --gradient_accumulation_steps=1 \
        --report_to "wandb" --tracker_project_name "gparmar_unpaired_h2z_cycle_debug_v2" \
        --enable_xformers_memory_efficient_attention --validation_steps 250 \
        --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1
    ```

- Additional optional flags:
    - `--enable_xformers_memory_efficient_attention`: Enable memory-efficient attention in the model.

### Step 3. Monitor the training progress
- You can monitor the training progress using the [Weights & Biases](https://wandb.ai/site) dashboard.

- The training script will visualizing the training batch, the training losses, and validation set L2, LPIPS, and FID scores (if specified).
    <div>
        <p align="center">
        <img src='../assets/examples/training_evaluation.png' align="center" width=800px>
        </p>
    </div>


- The model checkpoints will be saved in the `<output_dir>/checkpoints` directory.


### Step 4. Running Inference with the trained models

- You can run inference using the trained model using the following command:
    ```
    python src/inference_unpaired.py --model_path "output/cyclegan_turbo/my_horse2zebra/checkpoints/model_1001.pkl" \
        --input_image "data/my_horse2zebra/test_A/n02381460_20.jpg" \
        --prompt "picture of a zebra" --direction "a2b" \
        --output_dir "outputs" --image_prep "no_resize"
    ```

- The above command should generate the following output:
    <table>
    <tr>
    <th>Model Input</th>
    <th>Model Output</th>
    </tr>
    <tr>
    <td><img src='../assets/examples/my_horse2zebra_input.jpg' width="200px"></td>
    <td><img src='../assets/examples/my_horse2zebra_output.jpg' width="200px"></td>
    </tr>
    </table>

