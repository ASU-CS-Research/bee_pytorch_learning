from typing import List, Tuple

from torch import nn
from torch.utils.data import Dataset

import labelling_query_strategies.labelling_query_strategy


class NoLabeling(labelling_query_strategies.labelling_query_strategy.LabellingQueryStrategy):

    def choose_data_to_label(self, model: nn, filename_list: List[str], dataset: Dataset) -> List[Tuple[str, int]]:
        return []
