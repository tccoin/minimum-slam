import yaml
import os


class Params():
    def __init__(self, params):
        if type(params) is str:
            self.load(params)
        else:
            self.params = params

    def load(self, path) -> None:
        try:
            with open(os.path.expanduser(path), 'r') as param_file:
                self.params = yaml.safe_load(param_file)
        except:
            raise FileNotFoundError('Parameter file not found')

    def save(self, path) -> None:
        with open(os.path.expanduser(path), 'w') as param_file:
            yaml.dump(self.params, param_file)

    def __str__(self):
        return str(self.params)

    def __repr__(self):
        return str(self.params)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        raise TypeError('Params is read-only')

    def __delitem__(self, key):
        raise TypeError('Params is read-only')

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def __contains__(self, key):
        return key in self.params

    def __eq__(self, other):
        return self.params == other.params
