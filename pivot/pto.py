import numpy as np

from .transform import PivotSpace 

Point = np.ndarray

class Pto:
    def __init__(self, query:Point, transform: PivotSpace):
        self.trans = transform
        self.query = query

    def 
