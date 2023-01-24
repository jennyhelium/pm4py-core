from pm4py.statistics.variants.log import get as variants_module
from pm4py.objects.log.obj import EventLog, Trace, Event
from copy import copy

def keep_one_trace_per_variant(log, parameters=None):
    """
    Keeps only one trace per variant (does not matter for basic inductive miner)

    Parameters
    --------------
    log
        Log
    parameters
        Parameters of the algorithm

    Returns
    --------------
    new_log
        Log (with one trace per variant)
    """
    if parameters is None:
        parameters = {}

    new_log = EventLog()
    if log is not None:
        variants = variants_module.get_variants(log, parameters=parameters)
        for var in variants:
            curr_trace = variants[var][0]
            new_trace = Trace(attributes=copy(curr_trace.attributes))
            new_trace.attributes["@@num_traces"] = len(variants[var])
            for ev in curr_trace:
                new_trace.append(ev)
            new_log.append(new_trace)

    return new_log


def keep_only_one_attribute_per_event(log, attribute_key):
    """
    Keeps only one attribute per event

    Parameters
    ---------------
    log
        Event log
    attribute_key
        Attribute key
    """
    new_log = EventLog()
    if log is not None:
        for trace in log:
            new_trace = Trace()
            for ev in trace:
                new_trace.append(Event({attribute_key: ev[attribute_key]}))
            new_log.append(new_trace)

    return new_log
