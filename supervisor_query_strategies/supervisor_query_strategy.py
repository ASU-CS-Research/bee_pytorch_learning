import abc
from typing import List

import torch.nn as nn
from torch.utils.data import Dataset


class SupervisorQueryStrategy(abc.ABC):

    @abc.abstractmethod
    def query_data(self, model: nn, dataset: Dataset, response_size: int = 50) -> List[str]:
        pass
