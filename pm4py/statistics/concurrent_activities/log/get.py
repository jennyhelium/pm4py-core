from enum import Enum

from pm4py.objects.conversion.log import converter
from pm4py.objects.log.util import sorting
from pm4py.util import exec_utils, constants, xes_constants


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY


def apply(interval_log, parameters=None):
    """
    Gets the number of times for which two activities have been concurrent in the log

    Parameters
    --------------
    interval_log
        Interval event log
    parameters
        Parameters of the algorithm, including:
        - Parameters.ACTIVITY_KEY => activity key
        - Parameters.START_TIMESTAMP_KEY => start timestamp
        - Parameters.TIMESTAMP_KEY => complete timestamp

    Returns
    --------------
    ret_dict
        Dictionaries associating to a couple of activities (tuple) the number of times for which they have been
        executed in parallel in the log
    """
    if parameters is None:
        parameters = {}

    interval_log = converter.apply(interval_log, parameters=parameters)

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    start_timestamp_key = exec_utils.get_param_value(Parameters.START_TIMESTAMP_KEY, parameters,
                                                     xes_constants.DEFAULT_TIMESTAMP_KEY)

    ret_dict = {}
    for trace in interval_log:
        sorted_trace = sorting.sort_timestamp_trace(trace, start_timestamp_key)
        i = 0
        while i < len(sorted_trace):
            act1 = sorted_trace[i][activity_key]
            ts1 = sorted_trace[i][start_timestamp_key]
            tc1 = sorted_trace[i][timestamp_key]
            j = i + 1
            while j < len(sorted_trace):
                ts2 = sorted_trace[j][start_timestamp_key]
                tc2 = sorted_trace[j][timestamp_key]
                act2 = sorted_trace[j][activity_key]
                if max(ts1, ts2) <= min(tc1, tc2):
                    tup = (act1, act2)
                    if not tup in ret_dict:
                        ret_dict[tup] = 0
                    ret_dict[tup] = ret_dict[tup] + 1
                else:
                    break
                j = j + 1
            i = i + 1

    return ret_dict
