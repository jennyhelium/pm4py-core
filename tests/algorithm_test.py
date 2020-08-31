import unittest
from pm4py.objects.log.importer.xes import importer as xes_importer
import os


class AlgorithmTest(unittest.TestCase):
    def test_importing_xes(self):
        from pm4py.objects.log.importer.xes import importer as xes_importer
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"),
                                 variant=xes_importer.Variants.ITERPARSE)
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"),
                                 variant=xes_importer.Variants.LINE_BY_LINE)

    def test_importing_parquet(self):
        from pm4py.objects.log.importer.parquet import importer as parquet_importer
        df = parquet_importer.apply(os.path.join("input_data", "receipt.parquet"),
                                    variant=parquet_importer.Variants.PYARROW)
        df = parquet_importer.apply(os.path.join("input_data", "receipt.parquet"),
                                    variant=parquet_importer.Variants.FASTPARQUET)
        log = parquet_importer.import_log(os.path.join("input_data", "running-example.parquet"),
                                          variant=parquet_importer.Variants.PYARROW)
        log = parquet_importer.import_minimal_log(os.path.join("input_data", "running-example.parquet"),
                                                  variant=parquet_importer.Variants.PYARROW)

    def test_importing_csv(self):
        from pm4py.objects.log.importer.csv import importer as csv_importer
        df = csv_importer.import_dataframe_from_path(os.path.join("input_data", "running-example.csv"))
        df = csv_importer.import_dataframe_from_path_wo_timeconversion(
            os.path.join("input_data", "running-example.csv"))
        stream = csv_importer.apply(os.path.join("input_data", "running-example.csv"))
        stru = "case:concept:name,concept:name,time:timestamp\nA1,A,1970-01-01 01:00:00\n"
        df = csv_importer.import_dataframe_from_csv_string(stru)
        stream = csv_importer.import_log_from_string(stru)

    def test_hiearch_clustering(self):
        from pm4py.algo.clustering.trace_attribute_driven import algorithm as clust_algorithm
        log = xes_importer.apply(os.path.join("input_data", "receipt.xes"), variant=xes_importer.Variants.LINE_BY_LINE,
                                 parameters={xes_importer.Variants.LINE_BY_LINE.value.Parameters.MAX_TRACES: 50})
        # raise Exception("%d" % (len(log)))
        clust_algorithm.apply(log, "responsible", variant=clust_algorithm.Variants.VARIANT_DMM_VEC)

    def test_log_skeleton(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.log_skeleton import algorithm as lsk_discovery
        model = lsk_discovery.apply(log)
        from pm4py.algo.conformance.log_skeleton import algorithm as lsk_conformance
        conf = lsk_conformance.apply(log, model)

    def test_alignment(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net, im, fm = alpha_miner.apply(log)
        from pm4py.algo.conformance.alignments import algorithm as alignments
        aligned_traces = alignments.apply(log, net, im, fm, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
        aligned_traces = alignments.apply(log, net, im, fm, variant=alignments.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
        from pm4py.evaluation.replay_fitness import evaluator as rp_fitness_evaluator
        fitness = rp_fitness_evaluator.apply(log, net, im, fm, variant=rp_fitness_evaluator.Variants.ALIGNMENT_BASED)
        evaluation = rp_fitness_evaluator.evaluate(aligned_traces,
                                                   variant=rp_fitness_evaluator.Variants.ALIGNMENT_BASED)
        from pm4py.evaluation.precision import evaluator as precision_evaluator
        precision = precision_evaluator.apply(log, net, im, fm, variant=rp_fitness_evaluator.Variants.ALIGNMENT_BASED)

    def test_decomp_alignment(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net, im, fm = alpha_miner.apply(log)
        from pm4py.algo.conformance.decomp_alignments import algorithm as decomp_align
        aligned_traces = decomp_align.apply(log, net, im, fm, variant=decomp_align.Variants.RECOMPOS_MAXIMAL)

    def test_tokenreplay(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net, im, fm = alpha_miner.apply(log)
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        replayed_traces = token_replay.apply(log, net, im, fm, variant=token_replay.Variants.TOKEN_REPLAY)
        replayed_traces = token_replay.apply(log, net, im, fm, variant=token_replay.Variants.BACKWARDS)
        from pm4py.evaluation.replay_fitness import evaluator as rp_fitness_evaluator
        fitness = rp_fitness_evaluator.apply(log, net, im, fm, variant=rp_fitness_evaluator.Variants.TOKEN_BASED)
        evaluation = rp_fitness_evaluator.evaluate(replayed_traces, variant=rp_fitness_evaluator.Variants.TOKEN_BASED)
        from pm4py.evaluation.precision import evaluator as precision_evaluator
        precision = precision_evaluator.apply(log, net, im, fm,
                                              variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        from pm4py.evaluation.generalization import evaluator as generalization_evaluation
        generalization = generalization_evaluation.apply(log, net, im, fm,
                                                         variant=generalization_evaluation.Variants.GENERALIZATION_TOKEN)

    def test_evaluation(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net, im, fm = alpha_miner.apply(log)
        from pm4py.evaluation.simplicity import evaluator as simplicity
        simp = simplicity.apply(net)
        from pm4py.evaluation import evaluator as evaluation_method
        eval = evaluation_method.apply(log, net, im, fm)

    def test_playout(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net, im, fm = alpha_miner.apply(log)
        from pm4py.simulation.playout import simulator
        log2 = simulator.apply(net, im, fm)

    def test_tree_generation(self):
        from pm4py.simulation.tree_generator import simulator as tree_simulator
        tree1 = tree_simulator.apply(variant=tree_simulator.Variants.BASIC)
        tree2 = tree_simulator.apply(variant=tree_simulator.Variants.PTANDLOGGENERATOR)

    def test_alpha_miner_log(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net1, im1, fm1 = alpha_miner.apply(log, variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC)
        net2, im2, fm2 = alpha_miner.apply(log, variant=alpha_miner.Variants.ALPHA_VERSION_PLUS)
        from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
        dfg = dfg_discovery.apply(log)
        net3, im3, fm3 = alpha_miner.apply_dfg(dfg, variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC)

    def test_alpha_miner_dataframe(self):
        from pm4py.objects.log.adapters.pandas import csv_import_adapter
        df = csv_import_adapter.import_dataframe_from_path(os.path.join("input_data", "running-example.csv"))
        from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        net, im, fm = alpha_miner.apply(df, variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC)

    def test_tsystem(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.transition_system import algorithm as ts_system
        tsystem = ts_system.apply(log, variant=ts_system.Variants.VIEW_BASED)

    def test_inductive_miner(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.algo.discovery.inductive import algorithm as inductive_miner
        net, im, fm = inductive_miner.apply(log, variant=inductive_miner.Variants.IM)

    def test_performance_spectrum(self):
        log = xes_importer.apply(os.path.join("input_data", "running-example.xes"))
        from pm4py.statistics.performance_spectrum import algorithm as pspectrum
        ps = pspectrum.apply(log, ["register request", "decide"])
        from pm4py.objects.log.adapters.pandas import csv_import_adapter
        df = csv_import_adapter.import_dataframe_from_path(os.path.join("input_data", "running-example.csv"))
        ps = pspectrum.apply(df, ["register request", "decide"])

if __name__ == "__main__":
    unittest.main()
