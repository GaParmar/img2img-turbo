import subprocess
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def check_dependencies(yaml_file):
    yalm_source = read_yaml(yaml_file)
    if yalm_source is None:
        print("There are no dependencies.")
        return

    installed_packages = subprocess.check_output(['conda', 'list']).decode('utf-8').split('\n')[3:]
    installed_packages = [line.split()[0] for line in installed_packages if line.strip()]

    dependencies = yalm_source['dependencies']
    pip_dependencies = dependencies[1]['pip']
    missing_dependencies = []
    for dep in pip_dependencies:
        package_name = dep
        if dep.find("<") != -1:
            package_name = dep[:dep.find("<")].strip()
        elif dep.find(">") != -1:
            package_name = dep[:dep.find(">")].strip()
        elif dep.find("=") != -1:
            package_name = dep[:dep.find("=")].strip()
        elif dep.find("@") != -1:
            package_name = dep[:dep.find("@")].strip()
        try:
            subprocess.check_output(['pip', 'show', package_name])
        except subprocess.CalledProcessError:
            missing_dependencies.append(package_name)
            print(package_name, "is NOT installed.")
        print(package_name, "CHECKED")

    is_all_installed = False
    if not missing_dependencies:
        print("All dependencies are installed.")
        is_all_installed = True
    else:
        print("Missing dependencies:")
        for dep in missing_dependencies:
            print(dep)

    if is_all_installed:
        # 1. Test
        args = ['--model_name',
                'edge_to_image',
                '--input_image',
                'assets/examples/bird.png',
                '--prompt',
                'a blue bird',
                '--output_dir',
                './outputs']
        command_line = " ". join(['python',
                                    'src/inference_paired.py'] + args)
        # print(command_line)
        process = subprocess.Popen(['python',
                                    'src/inference_paired.py'] + args,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("1. Test passed.")
        else:
            print("1. Test NOT passed.")
            print(stdout.decode())
            print(stderr.decode())

        # 2. Test
        args = ['--model_name',
                'sketch_to_image_stochastic',
                '--input_image',
                './assets/examples/sketch_input.png',
                '--gamma',
                '0.4',
                '--prompt',
                'ethereal fantasy concept art of an asteroid. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy',
                '--output_dir',
                './outputs']
        command_line = " ". join(['python',
                                    'src/inference_paired.py'] + args)
        # print(command_line)
        process = subprocess.Popen(['python',
                                    'src/inference_paired.py'] + args,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("2. Test passed.")
        else:
            print("2. Test NOT passed.")
            print(stdout.decode())
            print(stderr.decode())

if __name__ == "__main__":
    yaml_file = "environment.yaml"
    check_dependencies(yaml_file)
