from setuptools import setup

setup(
    name='pm4py',
    version='1.0.5',
    packages=['pm4py', 'pm4py.algo', 'pm4py.algo.other', 'pm4py.algo.other.simple', 'pm4py.algo.other.simple.model',
              'pm4py.algo.other.simple.model.pandas', 'pm4py.algo.other.simple.model.pandas.versions',
              'pm4py.algo.other.simple.model.tracelog', 'pm4py.algo.other.simple.model.tracelog.versions',
              'pm4py.algo.other.simple.filtering', 'pm4py.algo.other.simple.filtering.pandas',
              'pm4py.algo.other.simple.filtering.pandas.versions', 'pm4py.algo.other.simple.filtering.tracelog',
              'pm4py.algo.other.simple.filtering.tracelog.versions', 'pm4py.algo.other.playout',
              'pm4py.algo.other.playout.versions', 'pm4py.algo.other.playout.data_structures', 'pm4py.algo.discovery',
              'pm4py.algo.discovery.dfg', 'pm4py.algo.discovery.dfg.utils', 'pm4py.algo.discovery.dfg.adapters',
              'pm4py.algo.discovery.dfg.adapters.pandas', 'pm4py.algo.discovery.dfg.versions',
              'pm4py.algo.discovery.alpha', 'pm4py.algo.discovery.alpha.utils', 'pm4py.algo.discovery.alpha.versions',
              'pm4py.algo.discovery.alpha.data_structures', 'pm4py.algo.discovery.causal',
              'pm4py.algo.discovery.causal.versions', 'pm4py.algo.discovery.inductive',
              'pm4py.algo.discovery.inductive.util', 'pm4py.algo.discovery.inductive.versions',
              'pm4py.algo.discovery.inductive.versions.dfg', 'pm4py.algo.discovery.inductive.versions.dfg.util',
              'pm4py.algo.discovery.inductive.versions.dfg.data_structures', 'pm4py.algo.filtering',
              'pm4py.algo.filtering.dfg', 'pm4py.algo.filtering.common', 'pm4py.algo.filtering.common.timestamp',
              'pm4py.algo.filtering.common.attributes', 'pm4py.algo.filtering.common.end_activities',
              'pm4py.algo.filtering.common.start_activities', 'pm4py.algo.filtering.pandas',
              'pm4py.algo.filtering.pandas.cases', 'pm4py.algo.filtering.pandas.paths',
              'pm4py.algo.filtering.pandas.variants', 'pm4py.algo.filtering.pandas.timestamp',
              'pm4py.algo.filtering.pandas.attributes', 'pm4py.algo.filtering.pandas.auto_filter',
              'pm4py.algo.filtering.pandas.end_activities', 'pm4py.algo.filtering.pandas.start_activities',
              'pm4py.algo.filtering.tracelog', 'pm4py.algo.filtering.tracelog.cases',
              'pm4py.algo.filtering.tracelog.paths', 'pm4py.algo.filtering.tracelog.variants',
              'pm4py.algo.filtering.tracelog.timestamp', 'pm4py.algo.filtering.tracelog.attributes',
              'pm4py.algo.filtering.tracelog.auto_filter', 'pm4py.algo.filtering.tracelog.end_activities',
              'pm4py.algo.filtering.tracelog.start_activities', 'pm4py.algo.conformance',
              'pm4py.algo.conformance.alignments', 'pm4py.algo.conformance.alignments.versions',
              'pm4py.algo.conformance.tokenreplay', 'pm4py.algo.conformance.tokenreplay.versions', 'pm4py.util',
              'pm4py.objects', 'pm4py.objects.log', 'pm4py.objects.log.util', 'pm4py.objects.log.adapters',
              'pm4py.objects.log.adapters.pandas', 'pm4py.objects.log.exporter', 'pm4py.objects.log.exporter.csv',
              'pm4py.objects.log.exporter.csv.versions', 'pm4py.objects.log.exporter.xes',
              'pm4py.objects.log.exporter.xes.versions', 'pm4py.objects.log.importer', 'pm4py.objects.log.importer.csv',
              'pm4py.objects.log.importer.csv.versions', 'pm4py.objects.log.importer.xes',
              'pm4py.objects.log.importer.xes.versions', 'pm4py.objects.petri', 'pm4py.objects.petri.common',
              'pm4py.objects.petri.exporter', 'pm4py.objects.petri.importer', 'pm4py.objects.conversion',
              'pm4py.objects.conversion.tree_to_petri', 'pm4py.objects.conversion.tree_to_petri.versions',
              'pm4py.objects.process_tree', 'pm4py.objects.process_tree.nodes_objects',
              'pm4py.objects.process_tree.nodes_threads', 'pm4py.objects.process_tree.trace_generation',
              'pm4py.objects.transition_system', 'pm4py.evaluation', 'pm4py.evaluation.precision',
              'pm4py.evaluation.precision.versions', 'pm4py.evaluation.simplicity',
              'pm4py.evaluation.simplicity.versions', 'pm4py.evaluation.generalization',
              'pm4py.evaluation.generalization.versions', 'pm4py.evaluation.replay_fitness',
              'pm4py.evaluation.replay_fitness.versions', 'pm4py.statistics', 'pm4py.statistics.traces',
              'pm4py.statistics.traces.common', 'pm4py.statistics.traces.pandas', 'pm4py.statistics.traces.tracelog',
              'pm4py.visualization', 'pm4py.visualization.dfg', 'pm4py.visualization.dfg.versions',
              'pm4py.visualization.common', 'pm4py.visualization.graphs', 'pm4py.visualization.graphs.util',
              'pm4py.visualization.graphs.versions', 'pm4py.visualization.petrinet',
              'pm4py.visualization.petrinet.util', 'pm4py.visualization.petrinet.common',
              'pm4py.visualization.petrinet.versions', 'pm4py.visualization.process_tree',
              'pm4py.visualization.process_tree.versions', 'tests.documentation_tests'],
    url='http://www.pm4py.org',
    license='GPL 3.0',
    author='PADS',
    author_email='pm4py@pads.rwth-aachen.de',
    description='Process Mining for Python',
    install_requires=[
        'numpy',
        'ciso8601',
        'cvxopt',
        'dataclasses',
        'flask',
        'flask-cors',
        'lxml',
        'graphviz',
        'pandas',
        'networkx==1.11',
        'scipy',
        'matplotlib'
    ]
)
