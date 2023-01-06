from enum import Enum
from typing import Any, Optional, Dict, Union, List, Tuple

import pandas as pd

from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.util import exec_utils
from pm4py.algo.transformation.log_to_features.variants import event_based, trace_based, temporal


class Variants(Enum):
    EVENT_BASED = event_based
    TRACE_BASED = trace_based
    TEMPORAL = temporal


def apply(log: Union[EventLog, pd.DataFrame, EventStream], variant: Any = Variants.TRACE_BASED,
          parameters: Optional[Dict[Any, Any]] = None) -> Tuple[Any, List[str]]:
    """
    Extracts the features from a log object

    Parameters
    ---------------
    log
        Event log
    variant
        Variant of the feature extraction to use:
        - Variants.EVENT_BASED => (default) extracts, for each trace, a list of numerical vectors containing for each
            event the corresponding features
        - Variants.TRACE_BASED => extracts for each trace a single numerical vector containing the features
            of the trace
        - Variants.TEMPORAL => extracts temporal features from the traditional event log

    Returns
    ---------------
    data
        Data to provide for decision tree learning
    feature_names
        Names of the features, in order
    """
    if parameters is None:
        parameters = {}

    return exec_utils.get_variant(variant).apply(log, parameters=parameters)
