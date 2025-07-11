import subprocess
import sys

if __name__ == "__main__":
    assert sys.argv[1] in ['cyclegan', 'pix2pix'], f'Unknown entrypoint, must be one of the following: {["cyclegan", "pix2pix"]}'
    call_cmd = ["accelerate", "launch", "--main_process_port", "29501", f"src/train_{sys.argv[1]}_turbo.py"]
    call_cmd.extend(sys.argv[2:])
    print(f'Running {call_cmd}')
    subprocess.run(call_cmd)
