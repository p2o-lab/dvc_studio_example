import yaml
import operator
from functools import reduce


def load_parameter(parameter_path):
    with open('params.yaml', 'r') as stream:
        param_dict = yaml.safe_load(stream)
        return reduce(operator.getitem, parameter_path, param_dict)
