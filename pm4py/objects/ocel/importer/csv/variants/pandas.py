from typing import Optional, Dict, Any

import pandas as pd

from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.ocel.util import extended_table


def apply(file_path: str, objects_path: str = None, parameters: Optional[Dict[Any, Any]] = None) -> OCEL:
    """
    Imports an object-centric event log from a CSV file, using Pandas as backend

    Parameters
    -----------------
    file_path
        Path to the object-centric event log
    objects_path
        Optional path to a CSV file containing the objects dataframe
    parameters
        Parameters of the algorithm

    Returns
    ------------------
    ocel
        Object-centric event log
    """
    if parameters is None:
        parameters = {}

    table = pd.read_csv(file_path)

    objects = None
    if objects_path is not None:
        objects = pd.read_csv(objects_path)

    return extended_table.get_ocel_from_extended_table(table, objects, parameters=parameters)
