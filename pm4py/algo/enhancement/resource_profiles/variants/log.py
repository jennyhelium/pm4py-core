from datetime import datetime
from enum import Enum
from typing import Union, Optional, Dict, Any, Tuple

import pytz

from pm4py.objects.conversion.log import converter
from pm4py.objects.log.log import EventLog, Event
from pm4py.util import exec_utils, constants, xes_constants


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    RESOURCE_KEY = constants.PARAMETER_CONSTANT_RESOURCE_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY


def get_dt_from_string(dt: Union[datetime, str]) -> datetime:
    """
    If the date is expressed as string, do the conversion to a datetime.datetime object

    Parameters
    -----------
    dt
        Date (string or datetime.datetime)

    Returns
    -----------
    dt
        Datetime object
    """
    if type(dt) is str:
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

    dt = dt.replace(tzinfo=pytz.utc)
    return dt


def distinct_activities(log: EventLog, t1: Union[datetime, str], t2: Union[datetime, str], r: str,
                        parameters: Optional[Dict[str, Any]] = None) -> int:
    """
    Number of distinct activities done by a resource in a given time interval [t1, t2)

    Metric RBI 1.1 in Pika, Anastasiia, et al.
    "Mining resource profiles from event logs." ACM Transactions on Management Information Systems (TMIS) 8.1 (2017): 1-30.

    Parameters
    -----------------
    log
        Event log
    t1
        Left interval
    t2
        Right interval
    r
        Resource

    Returns
    -----------------
    distinct_activities
        Distinct activities
    """
    if parameters is None:
        parameters = {}

    log = converter.apply(log, variant=converter.Variants.TO_EVENT_STREAM)

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    resource_key = exec_utils.get_param_value(Parameters.RESOURCE_KEY, parameters, xes_constants.DEFAULT_RESOURCE_KEY)

    t1 = get_dt_from_string(t1)
    t2 = get_dt_from_string(t2)

    log = [x for x in log if t1 <= x[timestamp_key] < t2 and x[resource_key] == r]
    return len(set(x[activity_key] for x in log))


def activity_frequency(log: EventLog, t1: Union[datetime, str], t2: Union[datetime, str], r: str, a: str,
                       parameters: Optional[Dict[str, Any]] = None) -> float:
    """
    Fraction of completions of a given activity a, by a given resource r, during a given time slot, [t1, t2),
    with respect to the total number of activity completions by resource r during [t1, t2)

    Metric RBI 1.3 in Pika, Anastasiia, et al.
    "Mining resource profiles from event logs." ACM Transactions on Management Information Systems (TMIS) 8.1 (2017): 1-30.

    Parameters
    -----------------
    log
        Event log
    t1
        Left interval
    t2
        Right interval
    r
        Resource
    a
        Activity

    Returns
    ----------------
    metric
        Value of the metric
    """
    if parameters is None:
        parameters = {}

    log = converter.apply(log, variant=converter.Variants.TO_EVENT_STREAM)

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    resource_key = exec_utils.get_param_value(Parameters.RESOURCE_KEY, parameters, xes_constants.DEFAULT_RESOURCE_KEY)

    t1 = get_dt_from_string(t1)
    t2 = get_dt_from_string(t2)
    log = [x for x in log if t1 <= x[timestamp_key] < t2 and x[resource_key] == r]
    total = len(log)

    log = [x for x in log if x[activity_key] == a]
    activity_a = len(log)

    return float(activity_a) / float(total) if total > 0 else 0.0


def activity_completions(log: EventLog, t1: Union[datetime, str], t2: Union[datetime, str], r: str,
                         parameters: Optional[Dict[str, Any]] = None) -> int:
    """
    The number of activity instances completed by a given resource during a given time slot.

    Metric RBI 2.1 in Pika, Anastasiia, et al.
    "Mining resource profiles from event logs." ACM Transactions on Management Information Systems (TMIS) 8.1 (2017): 1-30.

    Parameters
    -----------------
    log
        Event log
    t1
        Left interval
    t2
        Right interval
    r
        Resource

    Returns
    ----------------
    metric
        Value of the metric
    """
    if parameters is None:
        parameters = {}

    log = converter.apply(log, variant=converter.Variants.TO_EVENT_STREAM)

    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    resource_key = exec_utils.get_param_value(Parameters.RESOURCE_KEY, parameters, xes_constants.DEFAULT_RESOURCE_KEY)

    t1 = get_dt_from_string(t1)
    t2 = get_dt_from_string(t2)
    log = [x for x in log if t1 <= x[timestamp_key] < t2 and x[resource_key] == r]
    total = len(log)

    return total


def case_completions(log: EventLog, t1: Union[datetime, str], t2: Union[datetime, str], r: str,
                     parameters: Optional[Dict[str, Any]] = None) -> int:
    """
    The number of cases completed during a given time slot in which a given resource was involved.

    Metric RBI 2.2 in Pika, Anastasiia, et al.
    "Mining resource profiles from event logs." ACM Transactions on Management Information Systems (TMIS) 8.1 (2017): 1-30.

    Parameters
    -----------------
    log
        Event log
    t1
        Left interval
    t2
        Right interval
    r
        Resource

    Returns
    ----------------
    metric
        Value of the metric
    """
    if parameters is None:
        parameters = {}

    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    resource_key = exec_utils.get_param_value(Parameters.RESOURCE_KEY, parameters, xes_constants.DEFAULT_RESOURCE_KEY)
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, xes_constants.DEFAULT_TRACEID_KEY)

    t1 = get_dt_from_string(t1)
    t2 = get_dt_from_string(t2)

    last_eve = []
    stream = []
    for case in log:
        for i in range(len(case)):
            eve = Event({timestamp_key: case[i][timestamp_key], resource_key: case[i][resource_key],
                         case_id_key: case.attributes[case_id_key]})
            stream.append(eve)
            if i == len(case) - 1:
                last_eve.append(eve)

    last_eve = [x for x in last_eve if t1 <= x[timestamp_key] < t2]
    cases_last = set(x[case_id_key] for x in last_eve)

    stream = [x for x in stream if x[resource_key] == r]
    cases_res = set(x[case_id_key] for x in stream)

    return len(cases_last.intersection(cases_res))


def fraction_case_completions(log: EventLog, t1: Union[datetime, str], t2: Union[datetime, str], r: str,
                              parameters: Optional[Dict[str, Any]] = None) -> float:
    """
    The fraction of cases completed during a given time slot in which a given resource was involved with respect to the
    total number of cases completed during the time slot.

    Metric RBI 2.3 in Pika, Anastasiia, et al.
    "Mining resource profiles from event logs." ACM Transactions on Management Information Systems (TMIS) 8.1 (2017): 1-30.

    Parameters
    -----------------
    log
        Event log
    t1
        Left interval
    t2
        Right interval
    r
        Resource

    Returns
    ----------------
    metric
        Value of the metric
    """
    if parameters is None:
        parameters = {}

    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    resource_key = exec_utils.get_param_value(Parameters.RESOURCE_KEY, parameters, xes_constants.DEFAULT_RESOURCE_KEY)
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, xes_constants.DEFAULT_TRACEID_KEY)

    t1 = get_dt_from_string(t1)
    t2 = get_dt_from_string(t2)

    last_eve = []
    stream = []
    for case in log:
        for i in range(len(case)):
            eve = Event({timestamp_key: case[i][timestamp_key], resource_key: case[i][resource_key],
                         case_id_key: case.attributes[case_id_key]})
            stream.append(eve)
            if i == len(case) - 1:
                last_eve.append(eve)

    last_eve = [x for x in last_eve if t1 <= x[timestamp_key] < t2]
    cases_last = set(x[case_id_key] for x in last_eve)

    stream = [x for x in stream if x[resource_key] == r]
    cases_res = set(x[case_id_key] for x in stream)

    q1 = float(len(cases_last.intersection(cases_res)))
    q2 = float(len(cases_last))

    return q1 / q2 if q2 > 0 else 0.0


def __insert_start_from_previous_event(log: EventLog, parameters: Optional[Dict[str, Any]] = None) -> EventLog:
    """
    Inserts the start timestamp of an event set to the completion of the previous event in the case

    Parameters
    ---------------
    log
        interval log

    Returns
    ---------------
    log
        interval Log with the start timestamp for each event
    """
    if parameters is None:
        parameters = {}

    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    start_timestamp_key = exec_utils.get_param_value(Parameters.START_TIMESTAMP_KEY, parameters,
                                                     xes_constants.DEFAULT_START_TIMESTAMP_KEY)

    for trace in log:
        for i in range(1, len(trace)):
            trace[i][start_timestamp_key] = trace[i-1][timestamp_key]
        trace[0][start_timestamp_key] = trace[0][timestamp_key]

    return log


def __compute_workload(log: EventLog, resource: Optional[str] = None, activity: Optional[str] = None,
                       parameters: Optional[Dict[str, Any]] = None) -> Dict[Tuple, int]:
    """
    Computes the workload of resources/activities, corresponding to each event a number
    (number of concurring events)

    Parameters
    ---------------
    log
        event log
    resource
        (if provided) Resource on which we want to compute the workload
    activity
        (if provided) Activity on which we want to compute the workload

    Returns
    ---------------
    workload_dict
        Dictionary associating to each event the number of concurring events
    """
    if parameters is None:
        parameters = {}

    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    resource_key = exec_utils.get_param_value(Parameters.RESOURCE_KEY, parameters, xes_constants.DEFAULT_RESOURCE_KEY)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    start_timestamp_key = exec_utils.get_param_value(Parameters.START_TIMESTAMP_KEY, parameters, None)

    from pm4py.objects.log.util import sorting
    log = sorting.sort_timestamp(log, timestamp_key)
    from pm4py.objects.log.util import interval_lifecycle
    log = interval_lifecycle.to_interval(log, parameters=parameters)
    if start_timestamp_key is None:
        log = __insert_start_from_previous_event(log, parameters=parameters)
        start_timestamp_key = xes_constants.DEFAULT_START_TIMESTAMP_KEY
    events = converter.apply(log, variant=converter.Variants.TO_EVENT_STREAM)
    if resource is not None:
        events = [x for x in events if x[resource_key] == resource]
    if activity is not None:
        events = [x for x in events if x[activity_key] == activity]
    events = [(x[start_timestamp_key].timestamp(), x[timestamp_key].timestamp(), x[resource_key], x[activity_key]) for x
              in events]
    events = sorted(events)
    from intervaltree import IntervalTree, Interval
    tree = IntervalTree()
    ev_map = {}
    k = 0.000001
    for ev in events:
        tree.add(Interval(ev[0], ev[1] + k))
    for ev in events:
        ev_map[ev] = len(tree[ev[0]:ev[1] + k])
    return ev_map


def average_workload(log: EventLog, t1: Union[datetime, str], t2: Union[datetime, str], r: str,
                     parameters: Optional[Dict[str, Any]] = None) -> float:
    """
    The average number of activities started by a given resource but not completed at a moment in time.

    Metric RBI 2.4 in Pika, Anastasiia, et al.
    "Mining resource profiles from event logs." ACM Transactions on Management Information Systems (TMIS) 8.1 (2017): 1-30.

    Parameters
    -----------------
    log
        Event log
    t1
        Left interval
    t2
        Right interval
    r
        Resource

    Returns
    ----------------
    metric
        Value of the metric
    """
    if parameters is None:
        parameters = {}

    t2 = get_dt_from_string(t2).timestamp()

    ev_dict = __compute_workload(log, resource=r, parameters=parameters)
    ev_dict = {x: y for x, y in ev_dict.items() if x[0] < t2 and x[1] >= t2}
    num = 0.0
    den = 0.0
    for ev in ev_dict:
        workload = ev_dict[ev]
        duration = ev[1] - ev[0]
        num += workload*duration
        den += duration
    return num/den if den > 0 else 0.0
