import abc
from typing import List

import torch.nn as nn
from torch.utils.data import Dataset


class SupervisorQueryStrategy(abc.ABC):

    @abc.abstractmethod
    def query_data(self, results) -> List[int]:
        pass
