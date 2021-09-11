import toml
import subprocess

config = toml.load(open('config.toml'))
image = config['docker']['image']
command = ['bash', 'docker/clean.sh', image]
subprocess.check_call(command)
