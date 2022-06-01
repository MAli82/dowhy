"""This module provides functionality to estimate confidence intervals via bootstrapping the fitting and sampling.

Functions in this module should be considered experimental, meaning there might be breaking API changes in the future.
"""

from functools import partial
from typing import Union, Callable, Any, Dict

import numpy as np
import pandas as pd

from dowhy.gcm.cms import ProbabilisticCausalModel, InvertibleStructuralCausalModel, StructuralCausalModel
from dowhy.gcm.fitting_sampling import fit

# A convenience function when computing confidence intervals specifically for causal queries. This function
# bootstraps only sampling. Use bootstrap_training_and_sampling when you want to bootstrap training too.
#
# **Example usage:**
#
#     >>> gcm.fit(causal_model, data)
#     >>> scores_median, scores_intervals = gcm.confidence_intervals(
#     >>>     gcm.bootstrap_sampling(gcm.estimate_anomaly_scores, causal_model, target_node='Y'))
bootstrap_sampling = partial


def bootstrap_training_and_sampling(f: Callable[[Union[ProbabilisticCausalModel,
                                                       StructuralCausalModel,
                                                       InvertibleStructuralCausalModel], Any],
                                                Dict[Any, Union[np.ndarray, float]]],
                                    causal_model: Union[ProbabilisticCausalModel,
                                                        StructuralCausalModel,
                                                        InvertibleStructuralCausalModel],
                                    bootstrap_training_data: pd.DataFrame,
                                    bootstrap_data_subset_size_fraction: float = 0.75,
                                    *args, **kwargs):
    """A convenience function when computing confidence intervals specifically for causal queries. This function
    specifically bootstraps training *and* sampling.

    **Example usage:**

        >>> scores_median, scores_intervals = gcm.confidence_intervals(
        >>>     gcm.bootstrap_training_and_sampling(gcm.estimate_anomaly_scores,
        >>>                                         causal_model,
        >>>                                         bootstrap_training_data=data,
        >>>                                         target_node='Y'))

    :param f: The causal query to perform. A causal query is a function taking a graphical causal model as first
              parameter and an arbitrary number of remaining parameters. It must return a dictionary with
              attribution-like data.
    :param causal_model: A graphical causal model to perform the causal query on. It need not be fitted.
    :param bootstrap_training_data: The training data to use when fitting. A random subset from this data set is used
                                    in every iteration when calling fit.
    :param bootstrap_data_subset_size_fraction: The fraction defines the fractional size of the subset compared to
                                                the total training data.
    :param args: Args passed through verbatim to the causal queries.
    :param kwargs: Keyword args passed through verbatim to the causal queries.
    :return: A tuple containing (1) the median of causal query results and (2) the confidence intervals.
    """

    def snapshot():
        causal_model_copy = causal_model.clone()
        sampled_data = bootstrap_training_data.iloc[
            np.random.choice(bootstrap_training_data.shape[0],
                             int(bootstrap_training_data.shape[0] * bootstrap_data_subset_size_fraction),
                             replace=False)
        ]
        fit(causal_model_copy, sampled_data)
        return f(causal_model_copy, *args, **kwargs)

    return snapshot
