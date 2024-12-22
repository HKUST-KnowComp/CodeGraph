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
#
# Modifications made:
# Modified the task descriptions and few-shot examples for the following tasks using the CodeGraph method:
# - CycleCheck, EdgeExistence, NodeCount, NodeDegree, EdgeCount, ConnectedNodes



"""The graph tasks to be tried with LLMs."""

import random

import networkx as nx
import numpy as np

from codegraph import cg_graph_text_encoder as graph_text_encoder
import textwrap

class GraphTask:
    """The parent class for all the graph tasks."""

    def __init__(self):
        self.name = 'default'
        self.maximum_nnodes_cot_graph = 10

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        raise NotImplementedError()

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        raise NotImplementedError()


class CycleCheck(GraphTask):
    """The graph task to check if there is at least one cycle or not."""

    def __init__(self):
        super().__init__()
        self.name = 'cycle_check'
        self._task_graph_description = "Write a piece of Python code to return the answer in a variable 'ans'. Please enclose the code with # CODE START and # CODE END. Assume 'edges' and 'nodes' are empty lists (edges = [],nodes=[]) if not provided.\n"

    def get_nodes_string(self, name_dict, nnodes):
        node_string = ''
        for i in range(nnodes - 1):
            node_string += "'" + name_dict[i] + "',"
        node_string += "'" + name_dict[nnodes - 1] + "'"
        return node_string

    def get_edges_string(self, name_dict, edges):
        # Build the string with each edge on a new line and properly quoted
        edges_string = ',\n        '.join(f"('{name_dict[edge[0]]}', '{name_dict[edge[1]]}')" for edge in edges)
        return edges_string

    def generate_code(self, graph, encoding_method, name_dict):
        def create_node_string(name_dict, nnodes):
            node_string = ""
            for i in range(nnodes - 1):
                node_string += name_dict[i] + ", "
            node_string += name_dict[nnodes - 1]
            return node_string

        nodes_string = self.get_nodes_string(name_dict, len(graph.nodes()))
        edges_string = self.get_edges_string(name_dict, list(graph.edges()))
        graph_text = '{node: [] for node in nodes}'

        code = textwrap.dedent(f'''\
        # CODE START
        def has_cycle(graph):
            visited = set()
            for node in graph:
                if node not in visited:
                    if dfs(graph, node, visited, None):
                        return True
            return False
        def dfs(graph, node, visited, parent):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(graph, neighbor, visited, node):
                        return True
                elif neighbor != parent:
                    return True
            return False
        nodes = [{nodes_string}]
        edges = [{edges_string}]
        graph = {graph_text}
        for edge in edges:
             graph[edge[0]].append(edge[1])
        if has_cycle(graph):
            ans = "Has cycle."
        else:
            ans = "No cycle."
        # CODE END
        ''')
        return code

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        for ind, graph in enumerate(graphs):
            task_description = self._task_graph_description + 'Q: Is there a cycle in this graph?\n'
            question = (
                    self._task_graph_description + 'Q: Is there a cycle in this graph?\n'+graph_text_encoder.encode_graph(graph, encoding_method)
                    +'A: \n'
            )
            try:
                nx.find_cycle(graph)
                answer = 'Has cycle.'
            except nx.NetworkXNoCycle:
                answer = 'No cycle.'
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [],
            }
        return examples_dict

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        """Create a few shot example w or w/o cot for the graph graph."""
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = (
                self._task_graph_description + 'Q: Is there a cycle in this graph?\n' + graph_text_encoder.encode_graph(
            graph, encoding_method)+ 'A: \n'
        )
        answer = self.generate_code(graph, encoding_method, name_dict)
        return question + answer

    def choose_few_shot_examples(
            self,
            few_shots_dict,
            encoding_method,
            k=1,
    ):
        """Choose few shot examples for each algorithm."""
        pos_cycle_algorithms = ['er', 'ba', 'sbm', 'sfn', 'complete']
        neg_cycle_algorithms = ['star', 'path']
        few_shots_str = ''
        # choose k-1 shots for pos algorithms and one negative.
        positive_algorithms = random.choices(pos_cycle_algorithms, k=1)
        for positive_algorithm in positive_algorithms:
            example_list = few_shots_dict[(positive_algorithm, encoding_method)]
            few_shots_str += random.choice(example_list) + '\n'
        negative_algorithm = random.choice(neg_cycle_algorithms)
        example_list = few_shots_dict[(negative_algorithm, encoding_method)]
        few_shots_str += random.choice(example_list) + '\n'
        return few_shots_str


class EdgeExistence(GraphTask):
    """The graph task to check if an edge exist in a graph or not."""

    def __init__(self):
        super().__init__()
        self.name = 'edge_existence'
        self._task_insturction = "Write a piece of Python code to return the answer in a variable 'ans'. Please enclose the code with # CODE START and # CODE END. Assume 'edges' be an empty list (edges = []) if not provided.\n"

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

        for ind, graph in enumerate(graphs):
            source, target = random.sample(list(graph.nodes()), k=2)
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            task_description = 'Q: Is node %s connected to node %s? ' % (
                name_dict[source],
                name_dict[target],
            )
            question = self._task_insturction + task_description + question + 'A: '
            if ((source, target) in graph.edges()) or (
                    (target, source) in graph.edges()
            ):
                answer = 'True'
            else:
                answer = 'False'
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [source, target],
            }
        return examples_dict
    
    def get_edges_string(self, name_dict, edges):
        # Build the string with each edge on a new line and properly quoted
        edges_string = ',\n        '.join(f"('{name_dict[edge[0]]}', '{name_dict[edge[1]]}')" for edge in edges)
        return edges_string

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        source, target = random.sample(list(graph.nodes()), k=2)
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        task_description = 'Q: Is node %s connected to node %s? \n' % (
            name_dict[source],
            name_dict[target],
        )
        question = self._task_insturction + task_description + question + 'A: '

        answer = textwrap.dedent(f'''
        # CODE START
        source = '{name_dict[source]}'
        target = '{name_dict[target]}'
        edges = [{self.get_edges_string(name_dict, list(graph.edges()))}]
        def edge_existence(edges, source, target):
            edges_set = set()
            for u, v in edges:
                edges_set.add((u,v))
                edges_set.add((v,u))
            if (source, target) in edges_set: return True
            else: return False

        ans = edge_existence(edges,source,target)
        # CODE END
        ''')
        return question + answer


class NodeCount(GraphTask):
    """The graph task for finding number of nodes in a graph."""

    def __init__(self):
        super().__init__()
        self.name = 'node_count'
        self._task_graph_description = "Write a piece of Python code to return the answer in a variable 'ans'. Please enclose the code with # CODE START and # CODE END. Assume 'nodes' is an empty list (nodes=[]) if not provided.\nQ: How many nodes are in this graph? \n"

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        for ind, graph in enumerate(graphs):
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            self._task_description = self._task_graph_description
            question = self._task_description + question + 'A: '
            answer = ' %d.' % len(graph.nodes())
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': self._task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [],
            }
        return examples_dict

    def get_nodes_string(self, name_dict, nnodes):
        node_string = ''
        for i in range(nnodes - 1):
            node_string += "'" + name_dict[i] + "',"
        node_string += "'" + name_dict[nnodes - 1] + "'"
        return node_string

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        self._task_description = self._task_graph_description
        question = self._task_description + question + 'A: \n'
        # answer = '%d.' % len(graph.nodes())
        # if cot:
        #     answer += ' The nodes are %s.' % self.get_nodes_string(
        #         name_dict, len(graph.nodes())
        #     )

        formatted_node = self.get_nodes_string(name_dict, len(graph.nodes()))
        answer = textwrap.dedent(f'''\
        # CODE START
        nodes = [{formatted_node}]
        def count_nodes(nodes):
            return len(nodes)
        ans = count_nodes(nodes)
        # CODE END
        ''')
        return question + answer


class NodeDegree(GraphTask):
    """The graph task for finding degree of a node in a graph."""

    def __init__(self):
        super().__init__()
        self.name = 'node_degree'
        self._task_graph_description = "Write a piece of Python code to return the answer in a variable 'ans'. Please enclose the code with # CODE START and # CODE END. Assume 'edges' be an empty list (edges = []) if not provided.\n"

    def get_nodes_string(self, name_dict, nnodes):
        node_string = ''
        for i in range(nnodes - 1):
            node_string += "'" + name_dict[i] + "',"
        node_string += "'" + name_dict[nnodes - 1] + "'"
        return node_string

    def get_edges_string(self, name_dict, edges):
        # Build the string with each edge on a new line and properly quoted
        edges_string = ',\n        '.join(f"('{name_dict[edge[0]]}', '{name_dict[edge[1]]}')" for edge in edges)
        return edges_string


    # generate the code part in prompt,#CODE START...#CODE END
    def generate_code(self, graph, encoding_method, name_dict, source_node):
        def create_node_string(name_dict, nnodes):
            node_string = ""
            for i in range(nnodes - 1):
                node_string += name_dict[i] + ", "
            node_string += name_dict[nnodes - 1]
            return node_string
        nodes_string = self.get_nodes_string(name_dict, len(graph.nodes()))
        edges_string = self.get_edges_string(name_dict, list(graph.edges()))
        code = textwrap.dedent(f'''\
        # CODE START
        from collections import defaultdict
        from typing import Set, Dict, List, Tuple
        def get_adjacency_list(edges: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
            adjacency = defaultdict(set)
            for each_edge in edges:
                u, v = each_edge
                adjacency[u].add(v)
                adjacency[v].add(u)
            return adjacency
        def get_node_degree(target_node: str, adjacency_list: Dict[str, Set[str]]) -> int:
            return len(adjacency_list[target_node])
        edges = [{edges_string}]
        adjacency_list = get_adjacency_list(edges)
        target_node = '{str(name_dict[source_node])}'
        ans = get_node_degree(target_node, adjacency_list)
        # CODE END
        ''')
        return code

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        for ind, graph in enumerate(graphs):
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            source_node = random.sample(list(graph.nodes()), k=1)[0]
            task_description = self._task_graph_description +'Q: What is the degree of node %s? \n' % str(
                name_dict[source_node])

            question = task_description + question + 'A: \n'
            answer = '%d.' % graph.degree[source_node]
            # answer = code + 'Answer :' + answer + '\n'
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [source_node],
            }
        return examples_dict

    def get_edge_string(
            self, name_dict, graph, source_node
    ):
        """Gets a string identifying the edges a given node is connected to."""
        edge_string = ''
        target_edges = graph.edges(source_node)
        target_nodes = []
        for edge in target_edges:
            target_nodes.append(edge[1])
        if target_nodes:
            for i in range(len(target_nodes) - 1):
                edge_string += name_dict[target_nodes[i]] + ', '
            edge_string += 'and ' + name_dict[target_nodes[-1]]
        else:
            edge_string = 'no nodes'
        return edge_string

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        source_node = random.sample(list(graph.nodes()), k=1)[0]
        task_description = self._task_graph_description + 'Q: What is the degree of node %s? \n' % str(
            name_dict[source_node])
        question = task_description + question + 'A: \n'
        code = self.generate_code(graph, encoding_method, name_dict, source_node)
        answer = '%d.' % graph.degree[source_node]
        return question + code


class EdgeCount(GraphTask):
    """The graph task for finding number of edges in a graph."""

    def __init__(self):
        super().__init__()
        self.name = 'edge_count'
        self._task_description = "Write a piece of Python code to return the answer in a variable 'ans'. Please enclose the code with # CODE START and # CODE END. Assume 'edges' is an empty list (edges=[]) if not provided.\n"
        self._question_graph_description = 'Q: How many edges are in this graph? \n'
      #  self._task_description =''
    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        for ind, graph in enumerate(graphs):
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            self._question_description = self._question_graph_description
            question = self._task_description + self._question_description + question + 'A:'
            answer = ' %d.' % len(graph.edges())
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': self._task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [],
            }
        return examples_dict

    def get_edges_string(self, name_dict, edges):
        # Build the string with each edge on a new line and properly quoted
        edges_string = ',\n        '.join(f"('{name_dict[edge[0]]}', '{name_dict[edge[1]]}')" for edge in edges)
        return edges_string

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        self._question_description = self._question_graph_description

        question =self._task_description + self._question_description + question + 'A:'
        formatted_edges = self.get_edges_string(name_dict, list(graph.edges()))
        answer = textwrap.dedent(f'''
        # CODE START
        from typing import List, Tuple
        def count_edges(edges: List[Tuple[str, str]]) -> int:
            unique_edges = set()
            for u, v in edges:
                edge = tuple(sorted((u, v)))
                unique_edges.add(edge)
            return len(unique_edges)
        edges = [{formatted_edges}]
        ans = count_edges(edges)
        # CODE END
        ''')

        return question + answer


class ConnectedNodes(GraphTask):
    """The graph task for finding connected nodes to a given node in a graph."""

    def __init__(self):
        super().__init__()
        self.name = 'connected_nodes'
        self._task_graph_description = "Write a piece of Python code to return the answer in a variable 'ans'. Please enclose the code with # CODE START and # CODE END. Please create an adjacency list from 'edges', default 'edges' to an empty list (edges=[]) if not provided.\n"

    def get_nodes_string(self, name_dict, nnodes):
        node_string = ''
        for i in range(nnodes - 1):
            node_string += "'" + name_dict[i] + "',"
        node_string += "'" + name_dict[nnodes - 1] + "'"
        return node_string
    
    def get_edges_string(self, name_dict, edges):
        # Build the string with each edge on a new line and properly quoted
        edges_string = ',\n        '.join(f"('{name_dict[edge[0]]}', '{name_dict[edge[1]]}')" for edge in edges)
        return edges_string

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        for ind, graph in enumerate(graphs):
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            source_node = random.sample(list(graph.nodes()), k=1)[0]
            task_description = self._task_graph_description + 'Q: List all the nodes connected to %s in alphabetical order.\n'% name_dict[source_node] 
            question = task_description + question + 'A: \n' #question += task_description if you want to remove self.task_graph_description from the exemplar
            outgoing_edges = list(graph.edges(source_node))
            if outgoing_edges:
                answer = self.get_connected_nodes(outgoing_edges, name_dict) + '.'
            else:
                answer = ' No nodes.'
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [source_node],
            }
        return examples_dict

    def get_connected_nodes(
            self, edges, name_dict
    ):
        """Gets a string including all the nodes that are connected to source."""
        connected_nodes = []
        for edge in edges:
            connected_nodes.append(name_dict[edge[1]])
        connected_nodes_string = ''
        if connected_nodes:
            try:
                int(connected_nodes[0])
                connected_nodes_string = ', '.join(map(str, connected_nodes))
            except ValueError:
                # Check if these are not integers, sort
                connected_nodes_string = ', '.join(map(str, sorted(connected_nodes)))
        return connected_nodes_string

    # generate the code part in prompt,#CODE START...#CODE END
    def generate_code(self, graph, encoding_method,name_dict, source_node):
        def create_node_string(name_dict, nnodes):
            node_string = ""
            for i in range(nnodes - 1):
                node_string += name_dict[i] + ", "
            node_string += name_dict[nnodes - 1]
            return node_string
        nodes_string = self.get_nodes_string(name_dict, len(graph.nodes()))
        edges_string = self.get_edges_string(name_dict, list(graph.edges()))
        code = textwrap.dedent(f'''\
        # CODE START
        from collections import defaultdict
        from typing import Set, Dict, List, Tuple
        def get_adjacency_list(edges: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
            adjacency = defaultdict(set)
            for each_edge in edges:
                u, v = each_edge
                adjacency[u].add(v)
                adjacency[v].add(u)
            return adjacency
        def get_connected_nodes(target_node: str, adjacency_list: Dict[str, Set[str]]) -> str:
            if target_node in adjacency_list and adjacency_list[target_node]:
                connected_nodes = sorted(adjacency_list[target_node], key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
                return ', '.join(connected_nodes)
            else:
                return "No nodes"
        edges = [{edges_string}]
        adjacency_list = get_adjacency_list(edges)
        target_node = '{str(name_dict[source_node])}'
        ans = get_connected_nodes(target_node, adjacency_list)
        # CODE END
        ''')
        return code
    
    def get_edge_string(
            self, name_dict, graph, source_node
    ):
        """Gets a string identifying the edges a given node is connected to."""
        edge_string = ''
        target_edges = graph.edges(source_node)
        target_nodes = []
        for edge in target_edges:
            target_nodes.append(edge[1])
        if target_nodes:
            for i in range(len(target_nodes) - 1):
                edge_string += name_dict[target_nodes[i]] + ', '
            edge_string += 'and ' + name_dict[target_nodes[-1]]
        else:
            edge_string = 'no nodes'
        return edge_string


    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        source_node = random.sample(list(graph.nodes()), k=1)[0]
        task_description = self._task_graph_description + 'Q: List all the nodes connected to %s in alphabetical order.\n'% name_dict[source_node]
        question = task_description + question + 'A: \n'
        code = self.generate_code(graph, encoding_method, name_dict, source_node)
        outgoing_edges = list(graph.edges(source_node))
        answer = ''
        if outgoing_edges:
            answer = self.get_connected_nodes(outgoing_edges, name_dict) + '.'
        return question + code


class DisconnectedNodes(GraphTask):
    """The task for finding disconnected nodes for a given node in a graph."""

    def __init__(self):
        super().__init__()
        self.name = 'disconnected_nodes'

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        for ind, graph in enumerate(graphs):
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            source_node = random.sample(list(graph.nodes()), k=1)[0]
            task_description = (
                    'Q: List all the nodes that are not connected to %s in alphabetical'
                    ' order.\nA: '
                    % name_dict[source_node]
            )
            question += task_description
            outgoing_edges = list(graph.edges(source_node))
            answer = self.get_disconnected_nodes(
                source_node, outgoing_edges, name_dict, list(graph.nodes())
            )
            if not answer:
                answer = 'No nodes'

            answer += '.'
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [source_node],
            }
        return examples_dict

    def get_disconnected_nodes(
            self,
            source,
            edges,
            name_dict,
            all_nodes,
    ):
        """Gets a string with all the nodes that are not connected to source."""
        for edge in edges:
            if edge[1] in all_nodes:
                all_nodes.remove(edge[1])
        if source in all_nodes:
            all_nodes.remove(source)
        all_nodes_names = []
        for node in all_nodes:
            all_nodes_names.append(name_dict[node])
        # sorted operation should be different for integers vs strings.
        if all_nodes_names:
            try:
                int(all_nodes_names[0])
                for ind, value in enumerate(all_nodes_names):
                    all_nodes_names[ind] = int(value)
                all_nodes_names = sorted(all_nodes_names)
                for ind, value in enumerate(all_nodes_names):
                    all_nodes_names[ind] = str(value)
            except ValueError:
                pass
        return ', '.join(map(str, sorted(all_nodes_names)))

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        source_node = random.sample(list(graph.nodes()), k=1)[0]
        question += (
                'Q: List all the nodes that are not connected to %s in alphabetical'
                ' order.\nA: '
                % name_dict[source_node]
        )
        outgoing_edges = list(graph.edges(source_node))
        answer = ''
        disconnected_nodes_string = self.get_disconnected_nodes(
            source_node, outgoing_edges, name_dict, list(graph.nodes())
        )
        if outgoing_edges:
            if not disconnected_nodes_string:
                disconnected_nodes_string = 'No nodes'
            answer = disconnected_nodes_string + '.'
            if cot:
                answer += ' This is because there is not an edge from %s to %s.' % (
                    name_dict[source_node],
                    answer,
                )
            else:
                answer = ' No nodes.'
                if cot:
                    answer += (
                            ' This is because %s is connected to all nodes.'
                            % name_dict[source_node]
                    )
        return question + answer


class Reachability(GraphTask):
    """The graph task to check if there is a path from a source to target."""

    def __init__(self):
        super().__init__()
        self.name = 'reachability'

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

        for ind, graph in enumerate(graphs):
            source, target = random.sample(list(graph.nodes()), k=2)
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            task_description = 'Q: Is there a path from node %s to node %s?\nA: ' % (
                name_dict[source],
                name_dict[target],
            )
            question += task_description
            if nx.has_path(graph, source, target):
                answer = 'Yes.'
            else:
                answer = 'No.'
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [source, target],
            }
        return examples_dict

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        source, target = random.sample(list(graph.nodes()), k=2)
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        question += 'Q: Is there a path from node %s to node %s?\nA: ' % (
            name_dict[source],
            name_dict[target],
        )
        if nx.has_path(graph, source, target):
            answer = 'Yes.'
            if cot:
                path = nx.shortest_path(graph, source, target)
                explanation = ' Because'
                for i in range(len(path) - 1):
                    # The only edge or the non-last edges in the path.
                    if len(path) == 2 or i < len(path) - 2:
                        sep = ','
                    # The last edge in a path with more than one edge.
                    else:
                        sep = ', and'
                    explanation += '%s there is an edge from node %d to node %d' % (
                        sep,
                        path[i],
                        path[i + 1],
                    )
                explanation += ' .'
                answer += explanation
        else:
            answer = 'No.'
            if cot:
                answer += (
                        ' Because, there is no path connecting node %s to node %s based on'
                        ' the graph description.' % (name_dict[source], name_dict[target])
                )
        return question + answer


class ShortestPath(GraphTask):
    """The graph task to check if there is a path from a source to target."""

    def __init__(self):
        super().__init__()
        self.name = 'shortest_path'

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

        for ind, graph in enumerate(graphs):
            source, target = random.sample(list(graph.nodes()), k=1)
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            task_description = (
                    'Q: What is the length of the shortest path from node %s to node'
                    ' %s?\nA: '
                    % (
                        name_dict[source],
                        name_dict[target],
                    )
            )
            question += task_description
            try:
                path = nx.shortest_path(graph, source, target)
                answer = str(len(path) - 1) + '.'
            except nx.NetworkXNoPath:
                answer = 'There is no path from node %s to node %s.' % (
                    name_dict[source],
                    name_dict[target],
                )
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [source, target],
            }
        return examples_dict

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        source, target = random.sample(list(graph.nodes()), k=2)
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        question += (
                'Q: What is the length of the shortest path from node %s to node'
                ' %s?\nA: '
                % (
                    name_dict[source],
                    name_dict[target],
                )
        )
        if nx.has_path(graph, source, target):
            path = nx.shortest_path(graph, source, target)
            answer = str(len(path) - 1) + '.'
            if cot:
                explanation = ' Because'
                for i in range(len(path) - 1):
                    # The only edge or the non-last edges in the path.
                    if len(path) == 2 or i < len(path) - 2:
                        sep = ','
                    # The last edge in a path with more than one edge.
                    else:
                        sep = ', and'
                    explanation += '%s there is an edge from node %d to node %d' % (
                        sep,
                        path[i],
                        path[i + 1],
                    )
                explanation += ' .'
                answer += explanation
        else:
            answer = 'There is no path from node %s to node %s.' % (
                name_dict[source],
                name_dict[target],
            )
            if cot:
                answer += (
                        ' Because, there is no path connecting node %s to node %s based on'
                        ' the graph description.' % (name_dict[source], name_dict[target])
                )
        return question + answer


class TriangleCounting(GraphTask):
    """The graph task to count the number of triangles in a graph."""

    def __init__(self):
        super().__init__()
        self.name = 'triangle_counting'
        self._task_description = 'Q: How many triangles are in this graph?\nA: '

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        for ind, graph in enumerate(graphs):
            question = (
                    graph_text_encoder.encode_graph(graph, encoding_method)
                    + self._task_description
            )
            ntriangles = int(np.sum(list(nx.triangles(graph).values())) / 3)

            answer = '%i.' % ntriangles
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': self._task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [],
            }
        return examples_dict

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        """Create a few shot example w or w/o cot for the graph graph."""
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = (
                graph_text_encoder.encode_graph(graph, encoding_method)
                + self._task_description
        )
        triangles_dict = nx.triangles(graph)
        ntriangles = int(np.sum(list(triangles_dict.values())) / 3)

        if ntriangles > 0:
            answer = '%i.' % ntriangles
            if cot:
                ntriangles_cot = ''
                for key, value in triangles_dict.items():
                    if value > 0:
                        if value == 1:
                            ntriangles_cot += (
                                    'There is %i triangle including node %s as a vertex.\n'
                                    % (value, name_dict[key])
                            )
                        else:
                            ntriangles_cot += (
                                    'There are %i triangles including node %s as a vertex.\n'
                                    % (value, name_dict[key])
                            )
                ntriangles_cot += (
                        'Summing the number of triangles for all nodes and dividing them by'
                        ' three gives us %i triangles in total.' % ntriangles
                )
                answer += ntriangles_cot
        else:
            answer = '0.'
            if cot:
                ntriangles_cot = 'No three nodes form a triangle of edges.'
                answer += ntriangles_cot
        return question + answer


class MaximumFlow(GraphTask):
    """The graph task to compute the maximum flow from a source to a target."""

    def __init__(self):
        super().__init__()
        self.name = 'maximum_flow'

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

        for ind, graph in enumerate(graphs):
            graph = add_edge_weight(graph)
            source, target = random.sample(list(graph.nodes()), k=2)
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            task_description = (
                    'Q: What is the maximum capacity of the flow from node %s to node'
                    ' %s?\nA: ' % (name_dict[source], name_dict[target])
            )
            question += task_description
            maximum_flow_value = nx.maximum_flow(
                graph, source, target, capacity='weight'
            )[0]
            answer = str(maximum_flow_value) + '.'
            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(len(graph.nodes())),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                'node_ids': [source, target],
            }
        return examples_dict

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        graph = add_edge_weight(graph)
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        source, target = random.sample(list(graph.nodes()), k=2)
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        question += (
                'Q: What is the maximum capacity of the flow from node %s to'
                ' node %s?\nA: ' % (name_dict[source], name_dict[target])
        )
        flow_value, flow_dict = nx.maximum_flow(
            graph, source, target, capacity='weight'
        )
        answer = str(flow_value) + '.'
        if flow_value > 0:
            if cot:
                explanation = ' This is because of the following edges: '
                for edge, capacity in flow_dict.items():
                    for key, value in capacity.items():
                        if value > 0:
                            explanation += (
                                    'the edge from node %i to node %i with capacity %i, '
                                    % (
                                        edge,
                                        key,
                                        value,
                                    )
                            )
                explanation = explanation.strip()[:-1] + '.'
                answer += explanation
        else:
            if cot:
                answer += (
                        ' Because, there is no path connecting node %s to node %s based on'
                        ' the graph description.' % (name_dict[source], name_dict[target])
                )
        return question + answer


def has_edge_weights(graph):
    for _, _, data in graph.edges(data=True):
        if 'weight' not in data:
            return False
    return True


def add_edge_weight(graph):
    if has_edge_weights(graph):
        return graph
    else:
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['weight'] = random.randint(1, 10)
        return graph


class NodeClassification(GraphTask):
    """The graph task to classify a given node in the graph."""

    def __init__(self):
        super().__init__()
        self.name = 'node_classification'
        self.classes = [
            'soccer',
            'baseball',
            'tennis',
            'golf',
            'football',
            'surfing',
        ]

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
    ):
        classes = random.sample(list(self.classes), k=2)
        examples_dict = {}
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        for ind, graph in enumerate(graphs):
            question = graph_text_encoder.encode_graph(graph, encoding_method)
            nnodes = len(graph.nodes())
            # Sampling nnodes // 2 + 1 nodes.
            sampled_nodes = random.sample(
                list(graph.nodes(data=True)), k=nnodes // 2 + 1
            )
            # Adding the class of half of the nodes.
            for node_data in sampled_nodes[:-1]:
                node_class = classes[node_data[1]['block']]
                question += (
                        'Node ' + name_dict[node_data[0]] + ' likes ' + node_class + '.\n'
                )
            # Reserving the last sampled node for the question.
            task_description = 'Q: Does node %s like %s or %s?\nA: ' % (
                name_dict[sampled_nodes[-1][0]],
                classes[0],
                classes[1],
            )
            question += task_description
            answer = classes[sampled_nodes[-1][1]['block']]

            examples_dict[ind] = {
                'question': question,
                'answer': answer,
                'nnodes': str(nnodes),
                'nedges': str(len(graph.edges())),
                'task_description': task_description,
                'graph': graph,
                'algorithm': generator_algorithms[ind],
                # id of the last samples node
                'node_ids': [sampled_nodes[-1][0]],
            }

        return examples_dict

    def create_few_shot_example(
            self, graph, encoding_method, cot
    ):
        classes = random.sample(list(self.classes), k=2)
        name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
        question = graph_text_encoder.encode_graph(graph, encoding_method)
        nnodes = len(graph.nodes())
        sampled_nodes = random.sample(
            list(graph.nodes(data=True)), k=nnodes // 2 + 1
        )
        for node_data in sampled_nodes[:-1]:
            node_class = classes[node_data[1]['block']]
            question += (
                    'Node ' + name_dict[node_data[0]] + ' likes ' + node_class + '.\n'
            )
        task_description = 'Q: Does node %s like %s or %s?\nA: ' % (
            name_dict[sampled_nodes[-1][0]],
            classes[0],
            classes[1],
        )
        question += task_description
        answer = classes[sampled_nodes[-1][1]['block']]

        if cot:
            explanation = (
                    ' This is because most of the nodes that are connected to node %s'
                    ' likes %s.'
                    % (sampled_nodes[-1][0], classes[sampled_nodes[-1][1]['block']])
            )
            answer += explanation
        return question + answer
