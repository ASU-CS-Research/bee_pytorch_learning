from typing import List

import torch.nn as nn
from torch.utils.data import Dataset


class SupervisorQueryStrategy:

    def query_data(self, model: nn, filename_list: List[str], dataset: Dataset, response_size: int = 50) -> List[str]:
        raise NotImplementedError('SupervisorQueryStrategy is an abstract class.')
