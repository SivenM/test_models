import numpy as np


class Racall:

    def __init__(self):
        pass

    def calculate(self, tp, fn):
        return tp / (tp+fn)