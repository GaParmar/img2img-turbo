import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from google.cloud import aiplatform

from src import constants

TENSORBOARD_NAME = 'CycleGANTurboTensorboard'
TENSORBOARD_RESOURCE_NAME = 'projects/437403722141/locations/us-central1/tensorboards/397649375301468160'

def create_tensorboard():
    tensorboard = aiplatform.Tensorboard.create(
        display_name=TENSORBOARD_NAME, project=constants.PROJECT_ID, location=constants.LOCATION
    )
    return tensorboard.gca_resource.name

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run Vertex AI training job')
    parser.add_argument("image", type=str, help='Docker image SHA')
    parser.add_argument("--type", choices=['cyclegan', 'pix2pix'], required=True, help='Type of model to train')
    parser.add_argument("--experiment_name", type=str, required=True, help='Name of experiment')
    parser.add_argument("--dataset_name", type=str, required=True, help='Name of dataset to train on')
    parser.add_argument("--n_epochs", default=25, type=int, help='Number of training epochs to run')
    parser.add_argument("--pretrained_model_name_or_path", default='stabilityai/sd-turbo', type=str, help='Continue training from checkpoint')
    args = parser.parse_args()

    if not TENSORBOARD_RESOURCE_NAME:
        TENSORBOARD_RESOURCE_NAME = create_tensorboard() # Set TENSORBOARD_RESOURCE_NAME above in order to reuse the same tensorboard

    aiplatform.init(project=constants.PROJECT_ID, location=constants.LOCATION,
                    staging_bucket=constants.VERTEX_AI_BUCKET_NAME)

    job = aiplatform.CustomContainerTrainingJob(
        display_name='CycleGANTurboTraining',
        container_uri=f'gcr.io/{constants.PROJECT_ID}/{constants.IMAGE_NAME}@sha256:{args.image}',
        location=constants.LOCATION
    )

    job_args = [args.type,
                '--pretrained_model_name_or_path', args.pretrained_model_name_or_path,
                '--output_dir', f'/gcs/{constants.VERTEX_AI_BUCKET_NAME}/cyclegan_turbo_checkpoints/{args.experiment_name}',
                '--dataset_folder', f'/gcs/{constants.VERTEX_AI_BUCKET_NAME}/cyclegan_synthetic_to_real_silhouettes_dataset/{args.dataset_name}',
                '--max_train_steps', '25000',
                '--dataloader_num_workers', '8',
                '--train_batch_size', '2',
                '--gradient_accumulation_steps', '1',
                '--report_to', 'wandb',
                '--tracker_project_name', args.experiment_name,
                '--viz_freq', '25',
                '--enable_xformers_memory_efficient_attention']

    if args.type == 'cyclegan':
        job_args.extend(['--max_train_epochs', str(args.n_epochs),
                         '--validation_steps', '250',
                         '--learning_rate', '1e-5',
                         '--train_img_prep', 'resize_286_randomcrop_256x256_hflip',
                         '--val_img_prep', 'resize_256x256',
                         '--lambda_gan', '0.5',
                         '--lambda_idt', '1',
                         '--lambda_cycle', '1'])
    else:
        job_args.extend(['--num_training_epochs', str(args.n_epochs),
                         '--resolution', '512',
                         '--track_val_fid',
                         '--num_samples_eval', '500'])

    model = job.run(args=job_args,
                    replica_count=1,
                    machine_type='a2-highgpu-2g',
                    accelerator_type='NVIDIA_TESLA_A100',
                    accelerator_count=2,
                    boot_disk_size_gb=500,
                    service_account=constants.VERTEX_AI_SERVICE_ACCOUNT_EMAIL,
                    tensorboard=TENSORBOARD_RESOURCE_NAME)
