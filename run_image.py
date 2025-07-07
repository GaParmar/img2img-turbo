#!/usr/bin/env python3

import argparse
import os
import subprocess

from src import constants

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def build():
    print(f"Building Docker image")
    dockerfile_path = os.path.join(SCRIPT_DIR, "app", "Dockerfile")

    docker_build_cmd = ["docker", "build", "--build-arg", "WANDB_API_KEY=" + os.environ['WANDB_API_KEY'], "--tag", constants.IMAGE_NAME, "-f", dockerfile_path]
    docker_build_cmd.append(SCRIPT_DIR)
    print(' '.join(docker_build_cmd))
    subprocess.check_call(docker_build_cmd)

def push():
    build()

    print(f"Pushing Docker image {constants.IMAGE_NAME} to Google Container Registry")
    registry_image = f"gcr.io/{constants.PROJECT_ID}/{constants.IMAGE_NAME}"
    subprocess.check_call(["docker", "image", "tag", constants.IMAGE_NAME, registry_image])
    subprocess.check_call(["docker", "push", registry_image])

def run():
    build()

    PORT=8080
    image_key_path="/tmp/keys/google_key.json"
    local_key_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    docker_run_cmd = ["docker", "run"]
    docker_run_opt = ["-p", f"8080:{PORT}", "-e", f"PORT={PORT}", "-e", "K_SERVICE=dev",
                           "-e", "K_CONFIGURATION=dev", "-e", "K_REVISION=dev-00001",
                           "-e", f"GOOGLE_APPLICATION_CREDENTIALS={image_key_path}",
                           "-v", f"{local_key_path}:{image_key_path}:ro",
                           constants.IMAGE_NAME]
    if args.shell:
        docker_run_cmd.append("-it")
        docker_run_opt.append("sh")
    docker_run_cmd.extend(docker_run_opt)
    print(' '.join(docker_run_cmd))
    subprocess.check_call(docker_run_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image utilities')
    subparsers = parser.add_subparsers(help='Choose what you would like to do with this image. Default is build')
    parser.set_defaults(func=run)
    parser.set_defaults(shell=False)

    parser_build = subparsers.add_parser('build', help='Build the image locally')
    parser_build.set_defaults(func=build)

    parser_build = subparsers.add_parser('push', help='Push the image to Google Container Registry')
    parser_build.set_defaults(func=push)

    parser_run = subparsers.add_parser('run', help='Run the image locally')
    parser_run.set_defaults(func=run)
    parser_run.add_argument("-sh", "--shell", action="store_true", help="Run the image locally in an interactive shell")

    args = parser.parse_args()
    args.func()