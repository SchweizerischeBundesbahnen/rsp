# -*- coding: utf-8 -*-
import argparse

import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('extract', type=str,
                        choices=['conda_channels', 'conda_deps', 'pip_deps'],
                        help='which list to extract and print',
                        nargs=1)
    parser.add_argument('file', type=str, nargs=1, help='conda environment.yaml')
    args = parser.parse_args()
    with open(args.file[0]) as environment_yaml:
        content = environment_yaml.read()
    parsed = yaml.load(content, Loader=yaml.SafeLoader)
    conda_channels = parsed['channels']
    conda_dependencies = list(filter(lambda dep: isinstance(dep, str), parsed['dependencies']))
    pip_dependencies = next(filter(lambda dep: isinstance(dep, dict) and 'pip' in dep, parsed['dependencies']), None)
    parsed_dict = {
        'conda_channels': conda_channels,
        'conda_deps': conda_dependencies,
        'pip_deps': pip_dependencies['pip'] if pip_dependencies is not None else []
    }
    print("\n".join(parsed_dict[args.extract[0]]))
