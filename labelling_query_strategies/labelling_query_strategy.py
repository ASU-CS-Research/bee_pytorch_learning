from typing import List, Tuple
from torch.utils.data import Dataset
import torch.nn as nn


class LabellingQueryStrategy:

    def choose_data_to_label(self, model: nn, filename_list: List[str], dataset: Dataset) -> List[Tuple[str, int]]:
        raise NotImplementedError('LabellingQueryStrategy is an abstract class.')

