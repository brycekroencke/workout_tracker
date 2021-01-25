import json

#Managing hyperparameters such that they remain similar accross various testing programs

class get_config(dict):
    def __init__(self, *args, **kwargs):
        super(get_config, self).__init__(*args, **kwargs)
        self.__dict__ = self

def update_hyperparams(hyperparams):
    with open('../model/hyperparams.json', 'w+') as fp:
        json.dump(hyperparams, fp, indent=4)

def load_hyperparams(name):
    with open(name, 'r') as f:
        return json.load(f)

def get_hyperparam_obj():
    dict = load_hyperparams("../model/hyperparams.json")
    return get_config(dict)
