from pm4py.util import exec_utils
from enum import Enum
from pm4py.objects.ocel.obj import OCEL
from typing import Optional, Dict, Any
from pm4py.objects.ocel import constants as ocel_constants
from statistics import mean, median


class Parameters(Enum):
    EVENT_ID = ocel_constants.PARAM_EVENT_ID
    EVENT_ACTIVITY = ocel_constants.PARAM_EVENT_ACTIVITY
    OBJECT_ID = ocel_constants.PARAM_OBJECT_ID
    OBJECT_TYPE = ocel_constants.PARAM_OBJECT_TYPE


def apply(ocel: OCEL, parameters: Optional[Dict[Any, Any]] = None) -> Dict[str, Dict[str, float]]:
    """
    Provided statistics (mean, median, min, max) on the number of objects of a given type that are associated to events
    of a given activity.

    Parameters
    ---------------
    ocel
        Object-centric event log
    parameters
        Parameters of the algorithm, including:
        - Parameters.EVENT_ID => the event identifier
        - Parameters.EVENT_ACTIVITY => the activity
        - Parameters.OBJECT_ID => the object identifier
        - Parameters.OBJECT_TYPE => the object type

    Returns
    ---------------
    dictio
        Dictionary in which the first key is the activity, the second key is the object type,
        and the value is a dictionary containing the statistic for the given activity and object type.
    """
    if parameters is None:
        parameters = {}

    event_id = exec_utils.get_param_value(Parameters.EVENT_ID, parameters, ocel.event_id_column)
    event_activity = exec_utils.get_param_value(Parameters.EVENT_ACTIVITY, parameters, ocel.event_activity)
    object_id = exec_utils.get_param_value(Parameters.OBJECT_ID, parameters, ocel.object_id_column)
    object_type = exec_utils.get_param_value(Parameters.OBJECT_TYPE, parameters, ocel.object_type_column)

    activities = set(ocel.events[event_activity].unique())
    object_types = set(ocel.objects[object_type].unique())

    ret = {}

    for act in activities:
        if act not in ret:
            ret[act] = {}
        df = ocel.relations[ocel.relations[event_activity] == act]
        for ot in object_types:
            all_counts = list(df[df[object_type] == ot].groupby(event_id)[object_id].agg("count").to_dict().values())
            if all_counts:
                ret[act][ot] = {"min": min(all_counts), "max": max(all_counts), "mean": mean(all_counts),
                                "median": median(all_counts)}

    return ret
