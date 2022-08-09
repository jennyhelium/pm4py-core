import itertools
import sys
from abc import ABC
from collections import Counter
from itertools import product
from typing import Collection, Any, List, Optional, Generic
from typing import Tuple

from pm4py.algo.discovery.inductive.cuts.abc import Cut
from pm4py.algo.discovery.inductive.cuts.abc import T
from pm4py.algo.discovery.inductive.cuts import utils as cut_util
from pm4py.objects.dfg import util as dfu
from pm4py.objects.dfg.obj import DFG
from pm4py.objects.process_tree.obj import Operator, ProcessTree
from pm4py.util.compression.dtypes import UCL


class SequenceCut(Cut[T], ABC, Generic[T]):

    @classmethod
    def operator(cls) -> ProcessTree:
        return ProcessTree(operator=Operator.SEQUENCE)

    @classmethod
    def holds(cls, obj: T, dfg: DFG = None) -> Optional[List[Collection[Any]]]:
        '''
        This method finds a sequence cut in the dfg.
        Implementation follows function sequence on page 188 of
        "Robust Process Mining with Guarantees" by Sander J.J. Leemans (ISBN: 978-90-386-4257-4)

        Basic Steps:
        1. create a group per activity
        2. merge pairwise reachable nodes (based on transitive relations)
        3. merge pairwise unreachable nodes (based on transitive relations)
        4. sort the groups based on their reachability
        '''
        dfg = dfg if dfg is not None else obj if type(obj) is DFG else None
        alphabet = dfu.get_vertices(dfg)
        transitive_predecessors, transitive_successors = dfu.get_transitive_relations(dfg)
        groups = [{a} for a in alphabet]
        if len(groups) == 0:
            return None
        for a, b in product(alphabet, alphabet):
            if (b in transitive_successors[a] and a in transitive_successors[b]) or (
                    b not in transitive_successors[a] and a not in transitive_successors[b]):
                groups = cut_util.merge_groups_based_on_activities(a, b, groups)

        groups = list(sorted(groups, key=lambda g: len(
            transitive_predecessors[next(iter(g))]) + (len(alphabet) - len(transitive_successors[next(iter(g))]))))
        return groups if len(groups) > 1 else None


class StrictSequenceCut(SequenceCut[T], ABC, Generic[T]):

    @classmethod
    def _skippable(cls, p: int, dfg: DFG, start: Collection[Any], end: Collection[Any],
                   groups: List[Collection[Any]]) -> bool:
        """
        This method implements the function SKIPPABLE as defined on page 233 of
        "Robust Process Mining with Guarantees" by Sander J.J. Leemans (ISBN: 978-90-386-4257-4)
        The function is used as a helper function for the strict sequence cut detection mechanism, which detects
        larger groups of skippable activities.
        """
        for i, j in itertools.product(range(0, p), range(p + 1, len(groups))):
            for a, b in itertools.product(groups[i], groups[j]):
                if (a, b, f) in dfg.graph:
                    return True
        for i in range(p + 1, len(groups)):
            for a in groups[i]:
                if a in start:
                    return True
        for i in range(0, p):
            for a in groups[i]:
                if a in end:
                    return True
        return False

    @classmethod
    def holds(cls, obj: T, dfg: DFG = None) -> Optional[List[Collection[Any]]]:
        """
        This method implements the strict sequence cut as defined on page 233 of
        "Robust Process Mining with Guarantees" by Sander J.J. Leemans (ISBN: 978-90-386-4257-4)
        The function merges groups that together can be skipped.
        """
        dfg = dfg if dfg is not None else obj if type(obj) is DFG else None
        c = SequenceCut.apply(dfg)
        start = {a for (a, f) in dfg.start_activities}
        end = {a for (a, f) in dfg.end_activities}
        if c is not None:
            mf = [-1 * sys.maxsize if len(G.intersection(start)) > 0 else sys.maxsize for G in c]
            mt = [sys.maxsize if len(G.intersection(end)) > 0 else -1 * sys.maxsize for G in c]
            cmap = cls._construct_alphabet_cluster_map(c)
            for (a, b, f) in dfg.graph:
                mf[cmap[b]] = min(mf[cmap[b]], cmap[a])
                mt[cmap[a]] = max(mt[cmap[a]], cmap[b])

            for p in range(0, len(c)):
                if cls._skippable(p, dfg, start, end, c):
                    q = p - 1
                    while q >= 0 and mt[q] <= p:
                        c[p] = c[p].union(c[q])
                        c[q] = set()
                        q -= 1
                    q = p + 1
                    while q < len(mf) and mf[q] >= p:
                        c[p] = c[p].union(c[q])
                        c[q] = set()
                        q += 1
            return list(filter(lambda g: len(g) > 0, c))
        return None

    @classmethod
    def _construct_alphabet_cluster_map(cls, c: List[Collection[Any]]):
        map = dict()
        for i in range(0, len(c)):
            for a in c[i]:
                map[a] = i
        return map


class SequenceLogCut(SequenceCut[UCL]):

    @classmethod
    def project(cls, log: UCL, groups: List[Collection[Any]]) -> List[UCL]:
        logs = [list() for g in groups]
        for t in log:
            i = 0
            split_point = 0
            act_union = set()
            while i < len(groups):
                new_split_point = cls._find_split_point(
                    t, groups[i], split_point, act_union)
                trace_i = []
                j = split_point
                while j < new_split_point:
                    if t[j] in groups[i]:
                        trace_i.append(t[j])
                    j = j + 1
                logs[i].append(trace_i)
                split_point = new_split_point
                act_union = act_union.union(set(groups[i]))
                i = i + 1
        return logs

    @classmethod
    def _find_split_point(cls, t: List[Any], group: Collection[Any], start: int, ignore: Collection[Any]) -> int:
        least_cost = 0
        position_with_least_cost = start
        cost = 0
        i = start
        while i < len(t):
            if t[i] in group:
                cost = cost - 1
            elif t[i] not in ignore:
                cost = cost + 1

            if cost < least_cost:
                least_cost = cost
                position_with_least_cost = i + 1

            i = i + 1

        return position_with_least_cost


class StrictSequenceLogCut(StrictSequenceCut[UCL], SequenceLogCut):

    @classmethod
    def holds(cls, obj: T, dfg: DFG = None) -> Optional[List[Collection[Any]]]:
        return StrictSequenceCut.holds(obj, dfg)


class SequenceDFGCut(SequenceCut[DFG]):

    @classmethod
    def project(cls, dfg: DFG, groups: List[Collection[Any]]) -> Tuple[List[DFG], List[bool]]:
        start_activities = []
        end_activities = []
        activities = []
        dfgs = []
        skippable = []
        for g in groups:
            skippable.append(False)
        activities_idx = {}
        for gind, g in enumerate(groups):
            for act in g:
                activities_idx[act] = int(gind)
        i = 0
        while i < len(groups):
            to_succ_arcs = Counter()
            from_prev_arcs = Counter()
            if i < len(groups) - 1:
                for (a, b, f) in dfg.graph:
                    if a in groups[i] and b in groups[i + 1]:
                        to_succ_arcs[a] += f

            if i > 0:
                for (a, b, f) in dfg.graph:
                    if a in groups[i - 1] and b in groups[i]:
                        from_prev_arcs[b] += f

            if i == 0:
                start_activities.append({})
                for (a, f) in dfg.start_activities:
                    if a in groups[i]:
                        start_activities[i][a] = f
                    else:
                        j = i
                        while j < activities_idx[a]:
                            skippable[j] = True
                            j = j + 1
            else:
                start_activities.append(from_prev_arcs)

            if i == len(groups) - 1:
                end_activities.append({})
                for (a, f) in dfg.end_activities:
                    if a in groups[i]:
                        end_activities[i][a] = f
                    else:
                        j = activities_idx[a] + 1
                        while j <= i:
                            skippable[j] = True
                            j = j + 1
            else:
                end_activities.append(to_succ_arcs)

            activities.append({})
            act_count = dfu.get_vertex_frequencies(dfg)
            for a in groups[i]:
                activities[i][a] = act_count[a]
            dfgs.append({})
            for (a, b, f) in dfg.graph:
                if a in groups[i] and b in groups[i]:
                    dfgs[i][(a, b)] = f
            i = i + 1
        i = 0
        while i < len(dfgs):
            dfi = DFG()
            [dfi.graph.append((a, b, dfgs[i][(a, b)])) for (a, b) in dfgs[i]]
            [dfi.start_activities.append((a, start_activities[i][a])) for a in start_activities[i]]
            [dfi.end_activities.append((a, end_activities[i][a])) for a in end_activities[i]]
            dfgs[i] = dfi
            i = i + 1
        for (a, b, f) in dfg.graph:
            z = activities_idx[b]
            j = activities_idx[a] + 1
            while j < z:
                skippable[j] = False
                j = j + 1
        return dfgs, skippable
