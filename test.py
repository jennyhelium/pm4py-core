import pandas as pd

import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm
from pm4py.algo.conformance.alignments.petri_net import variants

df = pm4py.format_dataframe(pd.read_csv('pm4py/data/running_example.csv', sep=';'), case_id='case_id',
                            activity_key='activity',
                            timestamp_key='timestamp')
pn, im, fm = pm4py.discover_petri_net_inductive(df)
# pm4py.view_petri_net(pn, im, fm)

df_problems = pm4py.format_dataframe(pd.read_csv('pm4py/data/running_example_broken.csv', sep=';'),
                                     case_id='case:concept:name', activity_key='concept:name',
                                     timestamp_key='time:timestamp')

log = pm4py.convert_to_event_log(df_problems)

print(algorithm.create_data(log, pn, im, fm, variant=variants.a_star))
#print(algorithm.apply_all_heuristics(log, pn, im, fm, variant=variants.a_star))
# print(algorithm.apply(df_problems, pn, im, fm, variant=variants.a_star))
