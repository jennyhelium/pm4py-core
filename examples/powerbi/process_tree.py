if True:
    # ignore this part in true PowerBI executions
    import pandas as pd
    from pm4py.objects.log.util import dataframe_utils

    dataset = pd.read_csv("C:/running-example.csv")
    dataset = dataframe_utils.convert_timestamp_columns_in_df(dataset)

import pandas as pd

# this part is required because the dataframe provided by PowerBI has strings
dataset["time:timestamp"] = pd.to_datetime(dataset["time:timestamp"])
dataset = dataset.sort_values("time:timestamp")

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
tree = inductive_miner.apply_tree(dataset)

from pm4py.visualization.process_tree import visualizer
gviz = visualizer.apply(tree)
visualizer.matplotlib_view(gviz)
