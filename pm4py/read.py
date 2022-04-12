import warnings
from typing import Tuple, Dict, Optional

from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.obj import EventLog
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.process_tree.obj import ProcessTree

from pandas import DataFrame

INDEX_COLUMN = "@@index"


def read_xes(file_path: str, variant: str = "iterparse", **kwargs) -> DataFrame:
    """
    Reads an event log stored in XES format (see `xes-standard <https://xes-standard.org/>`_)
    Returns a table (``pandas.DataFrame``) view of the event log.

    :param file_path: file path of the event log (``.xes`` file) on disk

    .. code-block:: python3

        import pm4py

        log = pm4py.read_xes("<path_to_xes_file>")
    """
    from pm4py.objects.log.importer.xes import importer as xes_importer
    v = xes_importer.Variants.ITERPARSE
    if variant == "iterparse_20":
        v = xes_importer.Variants.ITERPARSE_20
    elif variant == "iterparse_mem_compressed":
        v = xes_importer.Variants.ITERPARSE_MEM_COMPRESSED
    elif variant == "line_by_line":
        v = xes_importer.Variants.LINE_BY_LINE
    log = xes_importer.apply(file_path, variant=v, parameters=kwargs)
    return log


def read_pnml(file_path: str) -> Tuple[PetriNet, Marking, Marking]:
    """
    Reads a Petri net object from a .pnmml file.
    The Petri net object returned is a triple containing the following objects:
    
    1. Petrinet Object, encoded as a ``PetriNet`` class
    #. Initial Marking
    #. Final Marking

    :rtype: ``Tuple[PetriNet, Marking, Marking]``
    :param file_path: file path of the Petri net model (``.pnml`` file) on disk

    .. code-block:: python3

        import pm4py

        pn = pm4py.read_pnml("<path_to_pnml_file>")
    """
    from pm4py.objects.petri_net.importer import importer as pnml_importer
    net, im, fm = pnml_importer.apply(file_path)
    return net, im, fm


def read_ptml(file_path: str) -> ProcessTree:
    """
    Reads a process tree object from a .ptml file

    :param file_path: file path of the process tree object on disk
 
    .. code-block:: python3

        import pm4py

        process_tree = pm4py.read_ptml("<path_to_ptml_file>")
    """
    from pm4py.objects.process_tree.importer import importer as tree_importer
    tree = tree_importer.apply(file_path)
    return tree


def read_dfg(file_path: str) -> Tuple[Dict[Tuple[str,str],int], Dict[str,int], Dict[str,int]]:
    """
    Reads a DFG object from a .dfg file.
    The DFG object returned is a triple containing the following objects:
    
    1. DFG Object, encoded as a ``Dict[Tuple[str,str],int]``, s.t. ``DFG[('a','b')]=k`` implies that activity ``'a'`` is directly followed by activity ``'b'`` a total of ``k`` times in the log
    #. Start activity dictionary, encoded as a ``Dict[str,int]``, s.t., ``S['a']=k`` implies that activity ``'a'`` is starting ``k`` traces in the event log
    #. End activity dictionary, encoded as a ``Dict[str,int]``, s.t., ``E['z']=k`` implies that activity ``'z'`` is ending ``k`` traces in the event log.

    :rtype: ``Tuple[Dict[Tuple[str,str],int], Dict[str,int], Dict[str,int]]``
    :param file_path: file path of the dfg model on disk
    

    .. code-block:: python3

       import pm4py

       dfg = pm4py.read_dfg("<path_to_dfg_file>")
    """
    from pm4py.objects.dfg.importer import importer as dfg_importer
    dfg, start_activities, end_activities = dfg_importer.apply(file_path)
    return dfg, start_activities, end_activities


def read_bpmn(file_path: str) -> BPMN:
    """
    Reads a BPMN model from a .bpmn file

    :param file_path: file path of the bpmn model

    .. code-block:: python3

        import pm4py

        bpmn = pm4py.read_bpmn('<path_to_bpmn_file>')

    """
    from pm4py.objects.bpmn.importer import importer as bpmn_importer
    bpmn_graph = bpmn_importer.apply(file_path)
    return bpmn_graph


def read_ocel(file_path: str, objects_path: Optional[str] = None) -> OCEL:
    """
    Reads an object-centric event log from a file (see: http://www.ocel-standard.org/).
    The ``OCEL`` object returned by this 

    :param file_path: file path of the object-centric event log
    :param objects_path: [Optional] file path from which the objects dataframe should be read

    .. code-block:: python3

        import pm4py

        ocel = pm4py.read_ocel("<path_to_ocel_file>")
    """
    if file_path.lower().endswith("csv"):
        from pm4py.objects.ocel.importer.csv import importer as csv_importer
        return csv_importer.apply(file_path, objects_path=objects_path)
    elif file_path.lower().endswith("jsonocel"):
        from pm4py.objects.ocel.importer.jsonocel import importer as jsonocel_importer
        return jsonocel_importer.apply(file_path)
    elif file_path.lower().endswith("xmlocel"):
        from pm4py.objects.ocel.importer.xmlocel import importer as xmlocel_importer
        return xmlocel_importer.apply(file_path)
