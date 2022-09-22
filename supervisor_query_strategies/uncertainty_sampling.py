from typing import List

from torch import no_grad, nn, max
from torch.utils.data import Dataset
import os

import supervisor_query_strategies.supervisor_query_strategy as sqs


class UncertaintySampling(sqs.SupervisorQueryStrategy):

    def query_data(self, model: nn, dataset: Dataset, response_size: int = 50) -> List[str]:
        with no_grad():
            conf_path = []
            for data in dataset:
                images, labels, paths = data
                images = images.float()
                # run the model on the test set to predict labels
                outputs = model(images)
                out = tuple(max(outputs.data, 1))
                conf_path.append((out[0].numpy()[0], paths[0], out[1].numpy()[0]))
            conf_path.sort(key=lambda a: a[0])
            # for j in range(len(conf_path) - 1, len(conf_path) - 11, -1):
            #     print(conf_path[j])
            return [os.path.abspath(item[1]) for item in conf_path]