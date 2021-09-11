import toml
import subprocess

config = toml.load(open('config.toml'))
base_image = config['docker']['base_image']
image = config['docker']['image']
command = ['bash', 'docker/setup.sh', base_image, image]
subprocess.check_call(command)
