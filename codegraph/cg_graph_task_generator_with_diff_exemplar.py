# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Modifications Copyright 2024 HKUST-KnowComp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications made:
# - Added new flags `_QUESTION_ALGORITHM` and `_EXEMPLAR_ALGORITHM` to handle distinct generation processes for questions and exemplars.
# - Modified the `few_shot` function to support different algorithms for questions and exemplars in task generation.

"""The graph tasks to be tried with LLMs."""

from collections.abc import Sequence
import os
import random

from absl import app
from absl import flags
import networkx as nx
import numpy as np

from codegraph import cg_graph_task as graph_task
from codegraph import cg_graph_task_utils_with_diff_exemplar as utils 

_TASK = flags.DEFINE_enum(
    'task',
    None,
    [
        'edge_existence',
        'node_degree',
        'node_count',
        'edge_count',
        'connected_nodes',
        'cycle_check',
        'disconnected_nodes',
        'reachability',
        'shortest_path',
        'maximum_flow',
        'triangle_counting',
        'node_classification',
    ],
    'The task to generate datapoints.',
    required=True,
)

_QUESTION_ALGORITHM = flags.DEFINE_enum(
    'question_algorithm',
    None,
    ['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path'],
    'The graph generator algorithm for the question.',
    required=True
)

_EXEMPLAR_ALGORITHM = flags.DEFINE_enum(
    'exemplar_algorithm',
    None,
    ['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path'],
    'The graph generator algorithm for the exemplars.',
    required=True
)

_TASK_DIR = flags.DEFINE_string(
    'task_dir', None, 'The directory to write tasks.', required=True
)
_GRAPHS_DIR = flags.DEFINE_string(
    'graphs_dir', None, 'The directory containing the graphs.', required=True
)
_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed',
    None,
    'The random seed to use for task generation.',
    required=True,
)
_K_SHOT = flags.DEFINE_integer(
    'k_shot', 
    1,  # default to 1-shot if not specified
    'Number of few-shot examples to include in the prompt'
)



TASK_CLASS = {
    'edge_existence': graph_task.EdgeExistence,
    'node_degree': graph_task.NodeDegree,
    'node_count': graph_task.NodeCount,
    'edge_count': graph_task.EdgeCount,
    'connected_nodes': graph_task.ConnectedNodes,
    'cycle_check': graph_task.CycleCheck,
    'disconnected_nodes': graph_task.DisconnectedNodes,
    'reachability': graph_task.Reachability,
    'shortest_path': graph_task.ShortestPath,
    'maximum_flow': graph_task.MaximumFlow,
    'triangle_counting': graph_task.TriangleCounting,
    'node_classification': graph_task.NodeClassification,
}

def few_shot(
    task,
    question_graphs,
    few_shot_graphs,
    question_algorithms,
    text_encoders,
    cot,
    bag,
    random_seed,
    k,
):
  """Creating few-shot, cot, or cot-bag examples for the given task.

  Args:
    task: the corresponding graph task.
    graphs: the list of graphs to use for the task.
    few_shot_graphs: the list of graphs to generate few shot examples for.
    algorithms: the algorithm used to generate the graphs.
    text_encoders: the encoders to use in the tasks.
    cot: whether to apply cot or not.
    bag: whether to apply build-a-graph method or not.
    random_seed: the random seed to use in the process.
  """
  random.seed(random_seed)
  few_shot_examples = utils.create_few_shot_task(
      task,
      question_graphs,
      question_algorithms,
      few_shot_graphs,
      text_encoders,
      cot=cot,
      bag=bag,
      random_seed=random_seed,
      k=k,
      
  )
  file_name = task.name
  if cot and bag:
    file_name += '_cot_bag_test.tfrecords'
  elif cot:
    file_name += '_cot_test.tfrecords'
  else:
    # file_name += '_few_shot_test.tfrecords'
    #file_name += '_cg_test.tfrecords'
    file_name += f'_cg_{k}_shot_test.tfrecords'

  utils.write_examples(
      few_shot_examples,
      os.path.join(_TASK_DIR.value, file_name),
  )


def generate_random_sbm_graph(random_state):
  # Sampling a small number as the probability of the two nodes in different
  # communities being connected.
  small_number = random.uniform(0, 0.05)
  # Sampling a large number as probability of the nodes in one community
  # being connected.
  large_number = random.uniform(0.6, 0.8)
  number_of_nodes = random.choice(np.arange(5, 20))
  sizes = [number_of_nodes // 2, number_of_nodes // 2]
  probs = [[large_number, small_number], [small_number, large_number]]
  return nx.stochastic_block_model(sizes, probs, seed=random_state)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  question_algorithm = _QUESTION_ALGORITHM.value
  exemplar_algorithm = _EXEMPLAR_ALGORITHM.value

  text_encoders = [
      'adjacency',
      'incident',
      'coauthorship',
      'friendship',
      'south_park',
      'got',
      'social_network',
      'politician',
      'expert',
  ]

  # Loading the graphs for questions
  question_graphs = utils.load_graphs(
      _GRAPHS_DIR.value,
      question_algorithm,
      'test',
  )
  question_generator_algorithms = [question_algorithm] * len(question_graphs)


  # Defining a task on the graphs
  task = TASK_CLASS[_TASK.value]()

  if isinstance(task, graph_task.NodeClassification):
    # The node classification task requires SBM graphs. As it's not possible to
    # write graphs with data (e.g., blocks data as in SBM graphs), we regenerate
    # graphs.

    random_state = np.random.RandomState(_RANDOM_SEED.value)
    print('Generating sbm graphs')
    graphs = [
        generate_random_sbm_graph(random_state) for _ in range(len(graphs))
    ]

  # Loading few-shot graphs.
    # Loading graphs for few-shot examples

  few_shot_graphs = utils.load_graphs(
      _GRAPHS_DIR.value,
      exemplar_algorithm,
      'train',
  )


  if isinstance(task, graph_task.NodeClassification):
    # The node classification task requires SBM graphs. As it's not possible to
    # write graphs with data (e.g., blocks data as in SBM graphs), we regenerate
    # graphs.
    random_state = np.random.RandomState(_RANDOM_SEED.value + 1)
    print('Generating few shot sbm graphs')
    few_shot_graphs = [
        generate_random_sbm_graph(random_state)
        for _ in range(len(few_shot_graphs))
    ]

  few_shot(
      task,
      question_graphs,
      few_shot_graphs,
      question_generator_algorithms,
      text_encoders,
      cot=False,
      bag=False,
      random_seed=_RANDOM_SEED.value,
      k=_K_SHOT.value,
  )


if __name__ == '__main__':
  app.run(main)
