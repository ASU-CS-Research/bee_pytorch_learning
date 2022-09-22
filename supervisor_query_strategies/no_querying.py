from typing import List

from torch import nn
from torch.utils.data import Dataset
import supervisor_query_strategies.supervisor_query_strategy as sqs

from helper_functions import log_message


class NoQuerying(sqs.SupervisorQueryStrategy):

    def query_data(self, model: nn, dataset: Dataset, response_size: int = 50) -> List[str]:
        log_message('No Querying selected, so no images will be returned.', 'INFO')
        return []
