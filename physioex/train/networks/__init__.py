import os

# import pkg_resources as pkg
import yaml

config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))

if not os.path.exists(config_file):
    raise FileNotFoundError(f"Network configuration file not found: {config_file}")


with open(config_file) as file:
    config = yaml.safe_load(file)
