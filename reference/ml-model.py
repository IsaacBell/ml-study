import numpy as np
from typing import TypeAlias, TypeVar, Generic

T = TypeVar('T')

class MlModel(object, Generic[T]):
    self.train_data: Generic[T] = []
    self.test_data: Generic[T] = []

    def __init__(self, type=None, name=None, dataset=np.array([])):
        pass

    # def __call__(self):

    def normalize():
        self.train_data

