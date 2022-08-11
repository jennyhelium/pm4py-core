from collections import Counter
from typing import Dict, Collection, Any, Tuple

import networkx as nx

from pm4py.objects.dfg.obj import DFG


def get_vertices(dfg: DFG) -> Collection[Any]:
    """
    Returns the vertices of the dfg

    :param dfg: input directly follows graph

    :rtype: ``Collection[Any]``
    """
    alphabet = set()
    [alphabet.update({a, b}) for (a, b, f) in dfg.graph]
    [alphabet.update({a}) for (a, f) in dfg.start_activities]
    [alphabet.update({a}) for (a, f) in dfg.end_activities]
    return alphabet


def get_outgoing_arcs(dfg: DFG) -> Dict[Any, Dict[Any, int]]:
    """
    Returns the outgoing arcs of the provided DFG graph.
    Returns a dictionary mapping each 'source' node onto its set of 'target' nodes and associated frequency.

    :param dfg: ``DFG`` object

    :rtype: ``Dict[str, Counter[str]]``

    """
    outgoing = {a: Counter() for a in get_vertices(dfg)}
    for (a, b, f) in dfg.graph:
        outgoing[a][b] = f if b not in outgoing[a] else outgoing[a][b] + f
    return outgoing


def get_incoming_arcs(dfg: DFG) -> Dict[Any, Dict[Any, int]]:
    """
    Returns the incoming arcs of the provided DFG graph.
    Returns a dictionary mapping each 'target' node onto its set of 'source' nodes and associated frequency.

    :param dfg: ``DFG`` object

    :rtype: ``Dict[str, Counter[str]]``

    """
    incoming = {a: Counter() for a in get_vertices(dfg)}
    for (a, b, f) in dfg.graph:
        incoming[b][a] = f if a not in incoming[b] else incoming[b][a] + f
    return incoming


def get_source_vertices(dfg: DFG) -> Collection[Any]:
    """
    Gets source vertices from a Directly-Follows Graph.
    Vertices are returned that have no incoming arcs

    :param dfg: ``DFG`` object

    :rtype: ``Collection[Any]``
    """
    starters = set()
    incoming = get_incoming_arcs(dfg)
    [starters.add(a) for a in incoming if len(incoming[a]) == 0]
    return starters


def get_sink_vertices(dfg: DFG) -> Collection[Any]:
    """
    Gets sink vertices from a Directly-Follows Graph.
    Vertices are returned that have no outgoing arcs

    :param dfg: ``DFG`` object

    :rtype: ``Collection[Any]``
    """
    ends = set()
    outgoing = get_outgoing_arcs(dfg)
    [ends.add(a) for a in outgoing if len(outgoing[a]) == 0]
    return ends


def get_transitive_relations(dfg: DFG) -> Tuple[Dict[Any, Collection[Any]], Dict[Any, Collection[Any]]]:
    '''
    Computes the full transitive relations in both directions (all activities reachable from a given activity and all
    activities that can reach the activity)

    :param dfg: ``DFG`` object

    :rtype: ``Tuple[Dict[Any, Collection[Any]], Dict[Any, Collection[Any]]] first argument maps an activity on all other
    activities that are able to reach the activity ('transitive pre set')
        second argument maps an activity on all other activities that it can reach (transitively) ('transitive post set')
    '''
    alph = get_vertices(dfg)
    pre = {a: set() for a in alph}
    post = {a: set() for a in alph}
    if len(dfg.graph) > 0:
        q = [(a, b) for (a, b, f) in dfg.graph]
        while len(q) > 0:
            s, t = q.pop(0)
            post[s].add(t)
            pre[t].add(s)
            post[s].update(post[t])
            pre[t].update(pre[s])
            for a, b, f in dfg.graph:
                if b == s and not post[s].issubset(post[a]):
                    post[a].update(post[s])
                    q.append((a, b))
                if a == t and not pre[t].issubset(pre[b]):
                    pre[b].update(pre[t])
                    q.append((a, b))
    return pre, post


def get_vertex_frequencies(dfg: DFG) -> Dict[Any, int]:
    '''
    Computes the number of times a vertex in the dfg is visited.
    The number equals the number of occurrences in the underlying log and is computed by summing up the incoming
    arc frequency and the number of starts in the vertex. The value is equal to the number of outgoing arcs combined
    with the number of endings of the vertex.
    '''
    c = Counter()
    for v in get_vertices(dfg):
        c[v] = 0
    for (a, b, f) in dfg.graph:
        c[v] += f
    for (a, f) in dfg.start_activities:
        c[v] += f
    return c


def as_nx_graph(dfg: DFG) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(get_vertices(dfg))
    for a, b, f in dfg.graph:
        nx_graph.add_edge(a, b)
    return nx_graph


def get_edges(dfg: DFG) -> Collection[Tuple[Any, Any]]:
    return {(a, b) for (a, b, f) in dfg.graph}
