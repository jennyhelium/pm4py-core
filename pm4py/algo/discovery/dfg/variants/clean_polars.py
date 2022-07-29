from enum import Enum

from pm4py.util import constants, exec_utils
from pm4py.util import xes_constants as xes_util
from enum import Enum
from typing import Optional, Dict, Any
import polars as pl
from pm4py.objects.dfg.obj import DFG
import time


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY


CONST_AUX_ACT = 'aux_act_'
CONST_AUX_CASE = 'aux_case_'
CONST_COUNT = 'count_'


def apply(log: pl.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> DFG:
    parameters = {} if parameters is None else parameters
    act_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes_util.DEFAULT_NAME_KEY)
    cid_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_ATTRIBUTE_GLUE)
    time_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY, parameters, xes_util.DEFAULT_TIMESTAMP_KEY)
    aux_act = CONST_AUX_ACT + str(time.time())
    aux_case = CONST_AUX_CASE + str(time.time())
    df = log[[cid_key, act_key, time_key]].clone()
    df = df.sort([cid_key, time_key])
    df = df[[cid_key, act_key]]
    df[aux_act] = df[act_key].shift(-1)
    df[aux_case] = df[cid_key].shift(-1)
    dfg = DFG()

    excl_starter = df[0, act_key]
    borders = df[(df[cid_key] != df[aux_case])]
    starters = list(filter(lambda d: d[aux_act] is not None, borders.groupby([aux_act]).count().to_dicts()))
    for d in starters:
        v = d['count'] + 1 if d[aux_act] == excl_starter else d['count']
        dfg.start_activities.append((d[aux_act],v))
    [dfg.end_activities.append((d[act_key], d['count'])) for d in filter(lambda d: d[act_key] is not None, borders.groupby([act_key]).count().to_dicts())]   
        
    [dfg.graph.append((d[act_key],d[aux_act],d['count'])) for d in df[(df[cid_key] == df[aux_case])].groupby([act_key, aux_act]).count().to_dicts()]
    
    return dfg
