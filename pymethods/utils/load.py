import numpy as np
import pathlib as pl


class Loadnpy:

    def __init__(self, path):
        self.path = pl.Path(path)

    def load(self, *args):
        if len(args) == 0:
            return np.load(self.path, allow_pickle=True)
        else:
            return np.load(self.path, allow_pickle=args[0])

    def load_item(self, *args):
        return self.load(*args).item()
