## Training with Unpaired Data (CycleGAN-turbo)
Here, we show how to train a CycleGAN-turbo model using unpaired data.
We will use the [horse2zebra dataset](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md) introduced by [CycleGAN](https://junyanz.github.io/CycleGAN/) as an example dataset.


### Step 1. Get the Dataset
- First download the horse2zebra dataset from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip).
    ```
    bash scripts/download_horse2zebra.sh
    ```
