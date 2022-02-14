import datetime
from typing import Optional, Tuple, Any, Collection, Union, List

import pandas as pd

from pm4py.objects.log.obj import EventLog, EventStream, Trace, Event
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.ocel.obj import OCEL
from pm4py.util import constants, xes_constants, pandas_utils

INDEX_COLUMN = "@@index"


def format_dataframe(df: pd.DataFrame, case_id: str = constants.CASE_CONCEPT_NAME,
                     activity_key: str = xes_constants.DEFAULT_NAME_KEY,
                     timestamp_key: str = xes_constants.DEFAULT_TIMESTAMP_KEY,
                     start_timestamp_key: str = xes_constants.DEFAULT_START_TIMESTAMP_KEY,
                     timest_format: Optional[str] = None) -> pd.DataFrame:
    """
    Give the appropriate format on the dataframe, for process mining purposes

    Parameters
    --------------
    df
        Dataframe
    case_id
        Case identifier column
    activity_key
        Activity column
    timestamp_key
        Timestamp column
    start_timestamp_key
        Start timestamp column
    timest_format
        Timestamp format that is provided to Pandas

    Returns
    --------------
    df
        Dataframe
    """
    if type(df) not in [pd.DataFrame, EventLog, EventStream]: raise Exception("the method can be applied only to a traditional event log!")

    from pm4py.objects.log.util import dataframe_utils
    if case_id not in df.columns:
        raise Exception(case_id + " column (case ID) is not in the dataframe!")
    if activity_key not in df.columns:
        raise Exception(activity_key + " column (activity) is not in the dataframe!")
    if timestamp_key not in df.columns:
        raise Exception(timestamp_key + " column (timestamp) is not in the dataframe!")
    if case_id != constants.CASE_CONCEPT_NAME:
        if constants.CASE_CONCEPT_NAME in df.columns:
            del df[constants.CASE_CONCEPT_NAME]
        df[constants.CASE_CONCEPT_NAME] = df[case_id]
    if activity_key != xes_constants.DEFAULT_NAME_KEY:
        if xes_constants.DEFAULT_NAME_KEY in df.columns:
            del df[xes_constants.DEFAULT_NAME_KEY]
        df[xes_constants.DEFAULT_NAME_KEY] = df[activity_key]
    if timestamp_key != xes_constants.DEFAULT_TIMESTAMP_KEY:
        if xes_constants.DEFAULT_TIMESTAMP_KEY in df.columns:
            del df[xes_constants.DEFAULT_TIMESTAMP_KEY]
        df[xes_constants.DEFAULT_TIMESTAMP_KEY] = df[timestamp_key]
    # makes sure that the timestamps column are of timestamp type
    df = dataframe_utils.convert_timestamp_columns_in_df(df, timest_format=timest_format)
    # drop NaN(s) in the main columns (case ID, activity, timestamp) to ensure functioning of the
    # algorithms
    df = df.dropna(subset={constants.CASE_CONCEPT_NAME, xes_constants.DEFAULT_NAME_KEY,
                           xes_constants.DEFAULT_TIMESTAMP_KEY}, how="any")
    # make sure the case ID column is of string type
    df[constants.CASE_CONCEPT_NAME] = df[constants.CASE_CONCEPT_NAME].astype("string")
    # make sure the activity column is of string type
    df[xes_constants.DEFAULT_NAME_KEY] = df[xes_constants.DEFAULT_NAME_KEY].astype("string")
    # set an index column
    df = pandas_utils.insert_index(df, INDEX_COLUMN)
    # sorts the dataframe
    df = df.sort_values([constants.CASE_CONCEPT_NAME, xes_constants.DEFAULT_TIMESTAMP_KEY, INDEX_COLUMN])
    # re-set the index column
    df = pandas_utils.insert_index(df, INDEX_COLUMN)
    # sets the properties
    if not hasattr(df, 'attrs'):
        # legacy (Python 3.6) support
        df.attrs = {}
    if start_timestamp_key in df.columns:
        df[xes_constants.DEFAULT_START_TIMESTAMP_KEY] = df[start_timestamp_key]
        df.attrs[constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY] = xes_constants.DEFAULT_START_TIMESTAMP_KEY
    df.attrs[constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = xes_constants.DEFAULT_NAME_KEY
    df.attrs[constants.PARAMETER_CONSTANT_TIMESTAMP_KEY] = xes_constants.DEFAULT_TIMESTAMP_KEY
    df.attrs[constants.PARAMETER_CONSTANT_GROUP_KEY] = xes_constants.DEFAULT_GROUP_KEY
    df.attrs[constants.PARAMETER_CONSTANT_TRANSITION_KEY] = xes_constants.DEFAULT_TRANSITION_KEY
    df.attrs[constants.PARAMETER_CONSTANT_RESOURCE_KEY] = xes_constants.DEFAULT_RESOURCE_KEY
    df.attrs[constants.PARAMETER_CONSTANT_CASEID_KEY] = constants.CASE_CONCEPT_NAME
    return df


def rebase(log_obj: Union[EventLog, EventStream, pd.DataFrame], case_id: str = constants.CASE_CONCEPT_NAME,
                     activity_key: str = xes_constants.DEFAULT_NAME_KEY,
                     timestamp_key: str = xes_constants.DEFAULT_TIMESTAMP_KEY,
                     start_timestamp_key: str = xes_constants.DEFAULT_START_TIMESTAMP_KEY):
    """
    Re-base the log object, changing the case ID, activity and timestamp attributes.

    Parameters
    -----------------
    log_obj
        Log object
    case_id
        Case identifier
    activity_key
        Activity
    timestamp_key
        Timestamp
    start_timestamp_key
        Start timestamp

    Returns
    -----------------
    rebased_log_obj
        Rebased log object
    """
    import pm4py

    if isinstance(log_obj, pd.DataFrame):
        return format_dataframe(log_obj, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key,
                                start_timestamp_key=start_timestamp_key)
    elif isinstance(log_obj, EventLog):
        log_obj = pm4py.convert_to_dataframe(log_obj)
        log_obj = format_dataframe(log_obj, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key,
                                   start_timestamp_key=start_timestamp_key)
        from pm4py.objects.conversion.log import converter
        return converter.apply(log_obj, variant=converter.Variants.TO_EVENT_LOG)
    elif isinstance(log_obj, EventStream):
        log_obj = pm4py.convert_to_dataframe(log_obj)
        log_obj = format_dataframe(log_obj, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key,
                                   start_timestamp_key=start_timestamp_key)
        return pm4py.convert_to_event_stream(log_obj)


def parse_process_tree(tree_string: str) -> ProcessTree:
    """
    Parse a process tree from a string

    Parameters
    ----------------
    tree_string
        String representing a process tree (e.g. '-> ( 'A', O ( 'B', 'C' ), 'D' )')
        Operators are '->': sequence, '+': parallel, 'X': xor choice, '*': binary loop, 'O' or choice

    Returns
    ----------------
    tree
        Process tree
    """
    from pm4py.objects.process_tree.utils.generic import parse
    return parse(tree_string)


def serialize(*args) -> Tuple[str, bytes]:
    """
    Serialize a PM4Py object into a bytes string

    Parameters
    -----------------
    args
        A PM4Py object, among:
        - an EventLog object
        - a Pandas dataframe object
        - a (Petrinet, Marking, Marking) tuple
        - a ProcessTree object
        - a BPMN object
        - a DFG, including the dictionary of the directly-follows relations, the start activities and the end activities

    Returns
    -----------------
    ser
        Serialized object (a tuple consisting of a string denoting the type of the object, and a bytes string
        representing the serialization)
    """
    from pm4py.objects.log.obj import EventLog
    from pm4py.objects.petri_net.obj import PetriNet
    from pm4py.objects.process_tree.obj import ProcessTree
    from pm4py.objects.bpmn.obj import BPMN
    from collections import Counter

    if type(args[0]) is EventLog:
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        return (constants.AvailableSerializations.EVENT_LOG.value, xes_exporter.serialize(*args))
    elif type(args[0]) is pd.DataFrame:
        from io import BytesIO
        buffer = BytesIO()
        args[0].to_parquet(buffer)
        return (constants.AvailableSerializations.DATAFRAME.value, buffer.getvalue())
    elif len(args) == 3 and type(args[0]) is PetriNet:
        from pm4py.objects.petri_net.exporter import exporter as petri_exporter
        return (constants.AvailableSerializations.PETRI_NET.value, petri_exporter.serialize(*args))
    elif type(args[0]) is ProcessTree:
        from pm4py.objects.process_tree.exporter import exporter as tree_exporter
        return (constants.AvailableSerializations.PROCESS_TREE.value, tree_exporter.serialize(*args))
    elif type(args[0]) is BPMN:
        from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
        return (constants.AvailableSerializations.BPMN.value, bpmn_exporter.serialize(*args))
    elif len(args) == 3 and (isinstance(args[0], dict) or isinstance(args[0], Counter)):
        from pm4py.objects.dfg.exporter import exporter as dfg_exporter
        return (constants.AvailableSerializations.DFG.value,
                dfg_exporter.serialize(args[0], parameters={"start_activities": args[1], "end_activities": args[2]}))


def deserialize(ser_obj: Tuple[str, bytes]) -> Any:
    """
    Deserialize a bytes string to a PM4Py object

    Parameters
    ----------------
    ser
        Serialized object (a tuple consisting of a string denoting the type of the object, and a bytes string
        representing the serialization)

    Returns
    ----------------
    obj
         A PM4Py object, among:
        - an EventLog object
        - a Pandas dataframe object
        - a (Petrinet, Marking, Marking) tuple
        - a ProcessTree object
        - a BPMN object
        - a DFG, including the dictionary of the directly-follows relations, the start activities and the end activities
    """
    if ser_obj[0] == constants.AvailableSerializations.EVENT_LOG.value:
        from pm4py.objects.log.importer.xes import importer as xes_importer
        return xes_importer.deserialize(ser_obj[1])
    elif ser_obj[0] == constants.AvailableSerializations.DATAFRAME.value:
        from io import BytesIO
        buffer = BytesIO()
        buffer.write(ser_obj[1])
        buffer.flush()
        return pd.read_parquet(buffer)
    elif ser_obj[0] == constants.AvailableSerializations.PETRI_NET.value:
        from pm4py.objects.petri_net.importer import importer as petri_importer
        return petri_importer.deserialize(ser_obj[1])
    elif ser_obj[0] == constants.AvailableSerializations.PROCESS_TREE.value:
        from pm4py.objects.process_tree.importer import importer as tree_importer
        return tree_importer.deserialize(ser_obj[1])
    elif ser_obj[0] == constants.AvailableSerializations.BPMN.value:
        from pm4py.objects.bpmn.importer import importer as bpmn_importer
        return bpmn_importer.deserialize(ser_obj[1])
    elif ser_obj[0] == constants.AvailableSerializations.DFG.value:
        from pm4py.objects.dfg.importer import importer as dfg_importer
        return dfg_importer.deserialize(ser_obj[1])


def get_properties(log):
    """
    Gets the properties from a log object

    Parameters
    -----------------
    log
        Log object

    Returns
    -----------------
    prop_dict
        Dictionary containing the properties of the log object
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]: return {}

    from copy import copy
    parameters = copy(log.properties) if hasattr(log, 'properties') else copy(log.attrs) if hasattr(log,
                                                                                                    'attrs') else {}
    return parameters


def set_classifier(log, classifier, classifier_attribute=constants.DEFAULT_CLASSIFIER_ATTRIBUTE):
    """
    Methods to set the specified classifier on an existing event log

    Parameters
    ----------------
    log
        Log object
    classifier
        Classifier that should be set:
        - A list of event attributes can be provided
        - A single event attribute can be provided
        - A classifier stored between the "classifiers" of the log object can be provided
    classifier_attribute
        The attribute of the event that should store the concatenation of the attribute values for the given classifier

    Returns
    ----------------
    log
        The same event log (methods acts inplace)
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]: raise Exception("the method can be applied only to a traditional event log!")

    if type(classifier) is list:
        pass
    elif type(classifier) is str:
        if type(log) is EventLog and classifier in log.classifiers:
            classifier = log.classifiers[classifier]
        else:
            classifier = [classifier]

    if type(log) is EventLog:
        for trace in log:
            for event in trace:
                event[classifier_attribute] = "+".join(list(event[x] for x in classifier))
        log.properties[constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = classifier_attribute
        log.properties[constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY] = classifier_attribute
    elif type(log) is pd.DataFrame:
        log[classifier_attribute] = log[classifier[0]]
        for i in range(1, len(classifier)):
            log[classifier_attribute] = log[classifier_attribute] + "+" + log[classifier[i]]
        log.attrs[constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = classifier_attribute
        log.attrs[constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY] = classifier_attribute
    else:
        raise Exception("setting classifier is not defined for this class of objects")

    return log


def parse_event_log_string(traces: Collection[str], sep: str = ",",
                           activity_key: str = xes_constants.DEFAULT_NAME_KEY,
                           timestamp_key: str = xes_constants.DEFAULT_TIMESTAMP_KEY,
                           case_id_key: str = xes_constants.DEFAULT_TRACEID_KEY) -> EventLog:
    """
    Parse a collection of traces expressed as strings
    (e.g., ["A,B,C,D", "A,C,B,D", "A,D"])
    to an event log

    Parameters
    ------------------
    traces
        Collection of traces expressed as strings
    sep
        Separator used to split the activities of a string trace
    activity_key
        The attribute that should be used as activity
    timestamp_key
        The attribute that should be used as timestamp
    case_id_key
        The attribute that should be used as case identifier

    Returns
    -----------------
    log
        Event log
    """
    log = EventLog()
    this_timest = 10000000
    for index, trace in enumerate(traces):
        activities = trace.split(sep)
        trace = Trace()
        trace.attributes[case_id_key] = str(index)
        for act in activities:
            event = Event({activity_key: act, timestamp_key: datetime.datetime.fromtimestamp(this_timest)})
            trace.append(event)
            this_timest = this_timest + 1
        log.append(trace)
    return log


def project_on_event_attribute(log: Union[EventLog, pd.DataFrame], attribute_key=xes_constants.DEFAULT_NAME_KEY) -> \
List[List[str]]:
    """
    Project the event log on a specified event attribute. The result is a list, containing a list for each case:
    all the cases are transformed to list of values for the specified attribute.

    Parameters
    --------------------
    log
        Event log / Pandas dataframe
    attribute_key
        The attribute to be used

    Returns
    --------------------
    projected_cases
        Projection on the given attribute (a list containing, for each case, a list of its values for the
        specified attribute).

        Example:

        pm4py.project_on_event_attribute(log, "concept:name")

        [['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 'examine thoroughly', 'check ticket', 'decide', 'pay compensation'],
        ['register request', 'check ticket', 'examine casually', 'decide', 'pay compensation'],
        ['register request', 'examine thoroughly', 'check ticket', 'decide', 'reject request'],
        ['register request', 'examine casually', 'check ticket', 'decide', 'pay compensation'],
        ['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 'check ticket', 'examine casually', 'decide', 'reinitiate request', 'examine casually', 'check ticket', 'decide', 'reject request'],
        ['register request', 'check ticket', 'examine thoroughly', 'decide', 'reject request']]
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]: raise Exception("the method can be applied only to a traditional event log!")

    output = []
    if pandas_utils.check_is_pandas_dataframe(log):
        pandas_utils.check_pandas_dataframe_columns(log)
        from pm4py.streaming.conversion import from_pandas
        it = from_pandas.apply(log, parameters={from_pandas.Parameters.ACTIVITY_KEY: attribute_key})
        for trace in it:
            output.append([x[xes_constants.DEFAULT_NAME_KEY] if xes_constants.DEFAULT_NAME_KEY is not None else None for x in trace])
    else:
        for trace in log:
            output.append([x[attribute_key] if attribute_key is not None else None for x in trace])
    return output


def sample_cases(log: Union[EventLog, pd.DataFrame], num_cases: int) -> Union[EventLog, pd.DataFrame]:
    """
    (Random) Sample a given number of cases from the event log.

    Parameters
    ---------------
    log
        Event log / Pandas dataframe
    num_cases
        Number of cases to sample

    Returns
    ---------------
    sampled_log
        Sampled event log (containing the specified amount of cases)
    """
    if isinstance(log, EventLog):
        from pm4py.objects.log.util import sampling
        return sampling.sample(log, num_cases)
    elif isinstance(log, pd.DataFrame):
        from pm4py.objects.log.util import dataframe_utils
        return dataframe_utils.sample_dataframe(log, parameters={"max_no_cases": num_cases})


def sample_events(log: Union[EventStream, OCEL], num_events: int) -> Union[EventStream, OCEL]:
    """
    (Random) Sample a given number of events from the event log.

    Parameters
    ---------------
    log
        Event stream / OCEL / Pandas dataframes
    num_events
        Number of events to sample

    Returns
    ---------------
    sampled_log
        Sampled event stream / OCEL / Pandas dataframes (containing the specified amount of events)
    """
    if isinstance(log, EventStream):
        from pm4py.objects.log.util import sampling
        return sampling.sample_stream(log, num_events)
    elif isinstance(log, OCEL):
        from pm4py.objects.ocel.util import sampling
        return sampling.sample_ocel_events(log, parameters={"num_entities": num_events})
    elif isinstance(log, pd.DataFrame):
        return log.sample(n=num_events)
