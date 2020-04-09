from os.path import dirname, join

from setuptools import setup

import pm4py


def read_file(filename):
    with open(join(dirname(__file__), filename)) as f:
        return f.read()


setup(
    name=pm4py.__name__,
    version=pm4py.__version__,
    description=pm4py.__doc__.strip(),
    long_description=read_file('README'),
    author=pm4py.__author__,
    author_email=pm4py.__author_email__,
    py_modules=[pm4py.__name__],
    include_package_data=True,
    packages=['pm4py', 'pm4py.algo', 'pm4py.algo.discovery', 'pm4py.algo.discovery.dfg',
              'pm4py.algo.discovery.dfg.utils', 'pm4py.algo.discovery.dfg.adapters',
              'pm4py.algo.discovery.dfg.adapters.pandas', 'pm4py.algo.discovery.dfg.versions',
              'pm4py.algo.discovery.alpha', 'pm4py.algo.discovery.alpha.utils', 'pm4py.algo.discovery.alpha.versions',
              'pm4py.algo.discovery.alpha.data_structures', 'pm4py.algo.discovery.causal',
              'pm4py.algo.discovery.causal.versions', 'pm4py.algo.discovery.inductive',
              'pm4py.algo.discovery.inductive.util', 'pm4py.algo.discovery.inductive.versions',
              'pm4py.algo.discovery.inductive.versions.dfg', 'pm4py.algo.discovery.inductive.versions.dfg.util',
              'pm4py.algo.discovery.inductive.versions.dfg.data_structures', 'pm4py.algo.discovery.heuristics',
              'pm4py.algo.discovery.heuristics.versions', 'pm4py.algo.discovery.log_skeleton',
              'pm4py.algo.discovery.log_skeleton.versions', 'pm4py.algo.discovery.transition_system',
              'pm4py.algo.discovery.transition_system.util', 'pm4py.algo.discovery.transition_system.versions',
              'pm4py.algo.filtering', 'pm4py.algo.filtering.dfg', 'pm4py.algo.filtering.log',
              'pm4py.algo.filtering.log.ltl', 'pm4py.algo.filtering.log.cases', 'pm4py.algo.filtering.log.paths',
              'pm4py.algo.filtering.log.variants', 'pm4py.algo.filtering.log.timestamp',
              'pm4py.algo.filtering.log.attributes', 'pm4py.algo.filtering.log.auto_filter',
              'pm4py.algo.filtering.log.end_activities', 'pm4py.algo.filtering.log.start_activities',
              'pm4py.algo.filtering.common', 'pm4py.algo.filtering.common.timestamp',
              'pm4py.algo.filtering.common.attributes', 'pm4py.algo.filtering.common.end_activities',
              'pm4py.algo.filtering.common.start_activities', 'pm4py.algo.filtering.pandas',
              'pm4py.algo.filtering.pandas.ltl', 'pm4py.algo.filtering.pandas.cases',
              'pm4py.algo.filtering.pandas.paths', 'pm4py.algo.filtering.pandas.variants',
              'pm4py.algo.filtering.pandas.timestamp', 'pm4py.algo.filtering.pandas.attributes',
              'pm4py.algo.filtering.pandas.auto_filter', 'pm4py.algo.filtering.pandas.end_activities',
              'pm4py.algo.filtering.pandas.start_activities', 'pm4py.algo.clustering',
              'pm4py.algo.clustering.trace_attribute_driven', 'pm4py.algo.clustering.trace_attribute_driven.dfg',
              'pm4py.algo.clustering.trace_attribute_driven.util',
              'pm4py.algo.clustering.trace_attribute_driven.variant',
              'pm4py.algo.clustering.trace_attribute_driven.merge_log',
              'pm4py.algo.clustering.trace_attribute_driven.leven_dist',
              'pm4py.algo.clustering.trace_attribute_driven.linkage_method', 'pm4py.algo.simulation',
              'pm4py.algo.simulation.playout', 'pm4py.algo.simulation.playout.versions',
              'pm4py.algo.simulation.playout.data_structures', 'pm4py.algo.simulation.montecarlo',
              'pm4py.algo.simulation.montecarlo.utils', 'pm4py.algo.simulation.montecarlo.versions',
              'pm4py.algo.simulation.tree_generator', 'pm4py.algo.simulation.tree_generator.versions',
              'pm4py.algo.conformance', 'pm4py.algo.conformance.alignments',
              'pm4py.algo.conformance.alignments.versions', 'pm4py.algo.conformance.tokenreplay',
              'pm4py.algo.conformance.tokenreplay.versions', 'pm4py.algo.conformance.tokenreplay.diagnostics',
              'pm4py.algo.conformance.log_skeleton', 'pm4py.algo.conformance.log_skeleton.versions',
              'pm4py.algo.conformance.decomp_alignments', 'pm4py.algo.conformance.decomp_alignments.versions',
              'pm4py.algo.enhancement', 'pm4py.algo.enhancement.sna', 'pm4py.algo.enhancement.sna.versions',
              'pm4py.algo.enhancement.sna.versions.log', 'pm4py.algo.enhancement.sna.versions.pandas',
              'pm4py.algo.enhancement.roles', 'pm4py.algo.enhancement.roles.common',
              'pm4py.algo.enhancement.roles.versions', 'pm4py.algo.enhancement.decision',
              'pm4py.algo.enhancement.comparison', 'pm4py.algo.enhancement.comparison.petrinet', 'pm4py.util',
              'pm4py.util.lp', 'pm4py.util.lp.util', 'pm4py.util.lp.versions', 'pm4py.util.dt_parsing',
              'pm4py.util.dt_parsing.versions', 'pm4py.objects', 'pm4py.objects.dfg', 'pm4py.objects.dfg.utils',
              'pm4py.objects.dfg.filtering', 'pm4py.objects.dfg.retrieval', 'pm4py.objects.log',
              'pm4py.objects.log.util', 'pm4py.objects.log.adapters', 'pm4py.objects.log.adapters.pandas',
              'pm4py.objects.log.exporter', 'pm4py.objects.log.exporter.csv', 'pm4py.objects.log.exporter.csv.versions',
              'pm4py.objects.log.exporter.xes', 'pm4py.objects.log.exporter.xes.versions',
              'pm4py.objects.log.exporter.parquet', 'pm4py.objects.log.exporter.parquet.versions',
              'pm4py.objects.log.importer', 'pm4py.objects.log.importer.csv', 'pm4py.objects.log.importer.csv.versions',
              'pm4py.objects.log.importer.xes', 'pm4py.objects.log.importer.xes.versions',
              'pm4py.objects.log.importer.parquet', 'pm4py.objects.log.importer.parquet.versions',
              'pm4py.objects.log.serialization', 'pm4py.objects.log.serialization.versions',
              'pm4py.objects.log.deserialization', 'pm4py.objects.log.deserialization.versions', 'pm4py.objects.petri',
              'pm4py.objects.petri.common', 'pm4py.objects.petri.exporter', 'pm4py.objects.petri.exporter.versions',
              'pm4py.objects.petri.importer', 'pm4py.objects.petri.importer.versions', 'pm4py.objects.conversion',
              'pm4py.objects.conversion.dfg', 'pm4py.objects.conversion.dfg.versions', 'pm4py.objects.conversion.log',
              'pm4py.objects.conversion.log.versions', 'pm4py.objects.conversion.process_tree',
              'pm4py.objects.conversion.process_tree.versions', 'pm4py.objects.conversion.heuristics_net',
              'pm4py.objects.conversion.heuristics_net.versions', 'pm4py.objects.process_tree',
              'pm4py.objects.heuristics_net', 'pm4py.objects.random_variables', 'pm4py.objects.random_variables.normal',
              'pm4py.objects.random_variables.uniform', 'pm4py.objects.random_variables.constant0',
              'pm4py.objects.random_variables.exponential', 'pm4py.objects.stochastic_petri',
              'pm4py.objects.transition_system', 'pm4py.streaming', 'pm4py.streaming.algo',
              'pm4py.streaming.algo.discovery', 'pm4py.streaming.algo.conformance', 'pm4py.streaming.stream',
              'pm4py.evaluation', 'pm4py.evaluation.precision', 'pm4py.evaluation.precision.versions',
              'pm4py.evaluation.simplicity', 'pm4py.evaluation.simplicity.versions', 'pm4py.evaluation.generalization',
              'pm4py.evaluation.generalization.versions', 'pm4py.evaluation.replay_fitness',
              'pm4py.evaluation.replay_fitness.versions', 'pm4py.statistics', 'pm4py.statistics.traces',
              'pm4py.statistics.traces.log', 'pm4py.statistics.traces.common', 'pm4py.statistics.traces.pandas',
              'pm4py.statistics.variants', 'pm4py.statistics.variants.log', 'pm4py.statistics.variants.pandas',
              'pm4py.statistics.attributes', 'pm4py.statistics.attributes.log', 'pm4py.statistics.attributes.common',
              'pm4py.statistics.attributes.pandas', 'pm4py.statistics.passed_time', 'pm4py.statistics.passed_time.log',
              'pm4py.statistics.passed_time.log.versions', 'pm4py.statistics.passed_time.pandas',
              'pm4py.statistics.passed_time.pandas.versions', 'pm4py.statistics.end_activities',
              'pm4py.statistics.end_activities.log', 'pm4py.statistics.end_activities.common',
              'pm4py.statistics.end_activities.pandas', 'pm4py.statistics.start_activities',
              'pm4py.statistics.start_activities.log', 'pm4py.statistics.start_activities.common',
              'pm4py.statistics.start_activities.pandas', 'pm4py.statistics.performance_spectrum',
              'pm4py.statistics.performance_spectrum.versions', 'pm4py.visualization', 'pm4py.visualization.dfg',
              'pm4py.visualization.dfg.versions', 'pm4py.visualization.sna', 'pm4py.visualization.sna.versions',
              'pm4py.visualization.common', 'pm4py.visualization.graphs', 'pm4py.visualization.graphs.util',
              'pm4py.visualization.graphs.versions', 'pm4py.visualization.petrinet',
              'pm4py.visualization.petrinet.util', 'pm4py.visualization.petrinet.common',
              'pm4py.visualization.petrinet.versions', 'pm4py.visualization.align_table',
              'pm4py.visualization.align_table.versions', 'pm4py.visualization.decisiontree',
              'pm4py.visualization.decisiontree.versions', 'pm4py.visualization.process_tree',
              'pm4py.visualization.process_tree.versions', 'pm4py.visualization.heuristics_net',
              'pm4py.visualization.heuristics_net.versions', 'pm4py.visualization.transition_system',
              'pm4py.visualization.transition_system.util', 'pm4py.visualization.transition_system.versions'],
    url='http://www.pm4py.org',
    license='GPL 3.0',
    install_requires=[
        'pyvis',
        'networkx',
        "matplotlib",
        'numpy',
        "ciso8601; python_version < '3.7'",
        'lxml',
        'graphviz',
        'pandas',
        'scipy',
        'scikit-learn',
        'pydotplus',
        'pulp',
        'pytz',
        'intervaltree'
    ],
    project_urls={
        'Documentation': 'http://www.pm4py.org',
        'Source': 'https://github.com/pm4py/pm4py-source',
        'Tracker': 'https://github.com/pm4py/pm4py-source/issues',
    }
)
