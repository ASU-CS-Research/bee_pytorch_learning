from typing import List

from torch import no_grad, nn
from torch.utils.data import Dataset
import os
import numpy as np
from helper_functions import log_message

import supervisor_query_strategies.supervisor_query_strategy as sqs


class MarginOfConfidence(sqs.SupervisorQueryStrategy):

    def query_data(self, model: nn, dataset: Dataset, response_size: int = 50) -> List[str]:
        # Begin by using the model on all the unlabeled images...
        log_message('Finding images to query the user for labels using Margin of Confidence Uncertainty Sampling.')
        log_message('Checking through every image and finding those with the least certainty, this may take a while...',
                    'DEBUG')
        with no_grad():
            conf_path = []
            for data in dataset:
                images, labels, paths = data
                images = images.float()
                log_message(f'Checking an image of size: {images.size()}', 'DEBUG')
                images = images.unsqueeze(0)
                # run the model on the test set to predict labels
                outputs = model(images)
                soft = nn.functional.softmax(outputs, dim=1)
                soft = soft.numpy().flatten()
                top = np.argpartition(soft, -2)[-2:]
                diff = soft[top][1] - soft[top][0]
                # Append confidence and filepath to 'conf_path'
                conf_path.append((diff, paths, top[1]))
            # Sort conf_path by confidence levels (lowest first)
            conf_path.sort(key=lambda a: a[0])
            # Return a list of filepaths equal in length to 'response_size'
            filepaths = [os.path.abspath(item[1]) for item in conf_path]
            filepaths = filepaths[:response_size]
            return filepaths
