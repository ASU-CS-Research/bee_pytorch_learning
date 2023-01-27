from typing import List

import numpy as np

import supervisor_labeling_strategies.supervisor_labeling_strategy as sqs


class UncertaintySampling(sqs.SupervisorLabelingStrategy):

    def query_data(self, results) -> List[int]:
        """
        Use the margin between the two top classes' probability and sort the results on this margin. The smallest margin
        is the most questionable data to the model, and the largest margin is the most certain.

        Args:
            results: results as ordered by the :mod:`ActiveLearner` method :meth:`ActiveLearner._test_accuracy`
        Returns:
            List[int]: List of integer indexes of the results, ordered based on the above description from most
              questionable to least questionable (the smallest margin to the largest margin).
        """
        # Since these are all unlabeled, the labels are irrelevant
        probabilities, labels = zip(*results)
        margins_and_indices = []
        for i, model_out in enumerate(probabilities):
            sorted_lst = np.sort(model_out)
            margin = abs(sorted_lst[-1] - sorted_lst[-2])
            margins_and_indices.append((margin, i))
        margins_and_indices = sorted(margins_and_indices, key=lambda entry: entry[0])
        margins, indices = zip(*margins_and_indices)
        return indices
