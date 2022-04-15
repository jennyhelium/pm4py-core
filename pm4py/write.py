__doc__ = """
The ``pm4py.write`` module contains all funcationality related to reading files/objects from disk.
"""

import warnings

from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.utils import __event_log_deprecation_warning
import pandas as pd
from typing import Union, Optional, Collection, Tuple
from pm4py.util import constants
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
from pm4py.objects.log.obj import XESExtension


def write_xes(log: Union[EventLog, pd.DataFrame], file_path: str, case_id_key: str = "case:concept:name", extensions: Optional[Collection[XESExtension]] = None, **kwargs) -> None:
    """
    Writes an event log to disk in the XES format (see `xes-standard <https://xes-standard.org/>`_)

    :param log: log object (``pandas.DataFrame``) that needs to be written to disk
    :param file_path: target file path of the event log (``.xes`` file) on disk
        
    .. code-block:: python3

        import pm4py

        pm4py.write_xes(log, '<path_to_export_to>')
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]: raise Exception("the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(log, case_id_key=case_id_key)

    parameters = {}
    for k, v in kwargs.items():
        parameters[k] = v
    parameters[constants.PARAMETER_CONSTANT_CASEID_KEY] = case_id_key
    parameters["extensions"] = extensions

    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    xes_exporter.apply(log, file_path, parameters=parameters)


def write_pnml(petri_net: PetriNet, initial_marking: Marking, final_marking: Marking, file_path: str) -> None:
    """
    Exports a (composite) Petri net object.

    Example:

    .. code-block:: python3

       import pm4py

       net, im, fm = pm4py.read_pnml("tests/input_data/running-example.pnml")
       pm4py.write_pnml(net, im, fm, "example.pnml")

    Parameters
    ------------
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    file_path
        Destination path

    Returns
    ------------
    void
    """
    from pm4py.objects.petri_net.exporter import exporter as petri_exporter
    petri_exporter.apply(petri_net, initial_marking, file_path, final_marking=final_marking)


def write_ptml(tree: ProcessTree, file_path: str) -> None:
    """
    Exports a process tree

    Example:

    .. code-block:: python3

       import pm4py

       process_tree = pm4py.read_ptml("tests/input_data/running-example.ptml")
       pm4py.write_ptml(process_tree, "example.ptml")

    Parameters
    ------------
    tree
        Process tree
    file_path
        Destination path

    Returns
    ------------
    void
    """
    from pm4py.objects.process_tree.exporter import exporter as tree_exporter
    tree_exporter.apply(tree, file_path)


def write_dfg(dfg: dict, start_activities: dict, end_activities: dict, file_path: str):
    """
    Exports a DFG

    Example:

    .. code-block:: python3

       import pm4py

       frequency_dfg = pm4py.read_dfg("tests/input_data/running-example.dfg")
       pm4py.write_dfg(frequency_dfg, "example.dfg")

    Parameters
    -------------
    dfg
        DFG
    start_activities
        Start activities
    end_activities
        End activities
    file_path
        Destination path

    Returns
    ------------
    void
    """
    from pm4py.objects.dfg.exporter import exporter as dfg_exporter
    dfg_exporter.apply(dfg, file_path,
                       parameters={dfg_exporter.Variants.CLASSIC.value.Parameters.START_ACTIVITIES: start_activities,
                                   dfg_exporter.Variants.CLASSIC.value.Parameters.END_ACTIVITIES: end_activities})


def write_bpmn(bpmn_graph: BPMN, file_path: str, enable_layout: bool = True):
    """
    Writes a BPMN to a file

    Example:

    .. code-block:: python3

       import pm4py

       bpmn_graph = pm4py.read_bpmn("tests/input_data/running-example.bpmn")
       pm4py.write_bpmn(bpmn_graph, "example.bpmn")

    Parameters
    ---------------
    bpmn_graph
        BPMN
    file_path
        Destination path
    enable_layout
        Enables the automatic layouting of the BPMN diagram (default: True)
    """
    if enable_layout:
        from pm4py.objects.bpmn.layout import layouter
        bpmn_graph = layouter.apply(bpmn_graph)
    from pm4py.objects.bpmn.exporter import exporter
    exporter.apply(bpmn_graph, file_path)


def write_ocel(ocel: OCEL, file_path: str, objects_path: str = None):
    """
    Stores the content of the object-centric event log to a file.
    Different formats are supported, including CSV (flat table), JSON-OCEL and XML-OCEL
    (described in the site http://www.ocel-standard.org/).

    Example:

    .. code-block:: python3

       import pm4py

       ocel = pm4py.read_ocel("tests/input_data/ocel/example_log.jsonocel")
       pm4py.write_ocel(ocel, "example.ocel")

    Parameters
    -----------------
    ocel
        Object-centric event log
    file_path
        Path at which the object-centric event log should be stored
    objects_path
        (Optional, only used in CSV exporter) Path where the objects dataframe should be stored
    """
    if file_path.lower().endswith("csv"):
        from pm4py.objects.ocel.exporter.csv import exporter as csv_exporter
        return csv_exporter.apply(ocel, file_path, objects_path=objects_path)
    elif file_path.lower().endswith("jsonocel"):
        from pm4py.objects.ocel.exporter.jsonocel import exporter as jsonocel_exporter
        return jsonocel_exporter.apply(ocel, file_path)
    elif file_path.lower().endswith("xmlocel"):
        from pm4py.objects.ocel.exporter.xmlocel import exporter as xmlocel_exporter
        return xmlocel_exporter.apply(ocel, file_path)
