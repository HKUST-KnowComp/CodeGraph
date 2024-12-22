# coding=utf-8
#Copyright 2024 HKUST-KnowComp Authors.
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

import os
from openai import AzureOpenAI
from openai import OpenAI

TEMPERATURE_GPT = 0.7
TEMPERATURE_LLAMA3 = 0.7
TEMPERATURE_MISTRAL = 1
TOP_P = 1 
FREQUENCY_PENALTY = 0 
PRESENCE_PENALTY = 0 
MAX_TOKEN = 8000

class Clients():
    def __init__(self, endpoint=None, api_key=None, api_version="2024-02-01", model_name='GPT35'):
        # Load API credentials from environment variables if not provided
        self.endpoint = endpoint or os.environ.get('AZURE_ENDPOINT')
        self.api_key = api_key or os.environ.get('AZURE_API_KEY')
        self.api_version = api_version
        
        if model_name == 'GPT35':
            if not self.endpoint or not self.api_key:
                raise ValueError("Azure endpoint and API key must be provided for GPT35 model.")

            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
            self.model = "GPT35"
            self.temperature = TEMPERATURE_GPT

        elif model_name == 'Llama_3_70B':
            self.api_key = api_key or os.environ.get('DEEPINFRA_API_KEY')
            self.base_url = os.environ.get('DEEPINFRA_BASE_URL', "https://api.deepinfra.com/v1/openai")

            if not self.api_key:
                raise ValueError("API key must be provided for Llama models.")

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.model = "meta-llama/Meta-Llama-3-70B-Instruct"
            self.temperature = TEMPERATURE_LLAMA3

        elif model_name.startswith('Mistral_8x'):
            self.api_key = api_key or os.environ.get('DEEPINFRA_API_KEY')
            self.base_url = os.environ.get('DEEPINFRA_BASE_URL', "https://api.deepinfra.com/v1/openai")

            if not self.api_key:
                raise ValueError("API key must be provided for Mistral models.")

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            if model_name == 'Mistral_8x7B':
                self.model = "mistralai/Mistral-8x7B-Instruct-v0.1"
            elif model_name == 'Mistral_8x22B':
                self.model = "mistralai/Mistral-8x22B-Instruct-v0.1"
            self.temperature = TEMPERATURE_MISTRAL

        else:
            raise ValueError(f"Model {model_name} not recognized.")

        self.frequency_penalty = FREQUENCY_PENALTY
        self.presence_penalty = PRESENCE_PENALTY
        self.top_p = TOP_P
        self.basic_text = ""
        self.task_specific_message_text = ""
        self.formatted_constrain_text = ""
        self.message_text = []
        self.max_token = MAX_TOKEN 

    def prompt_selection(self, prompt_method: str) -> None:
        assert prompt_method in ['few_shot', 'cot', 'zero_shot', 'cg'], NotImplementedError('The given prompt method hasn\'t implemented. Please double check')
        self.prompt_method = prompt_method

    def task_selection(self, task: str, text_enc: str) -> None:
        assert task in ['edge_existence', 'node_degree', 'node_count', 'edge_count', 'connected_nodes', 'cycle_check', 'disconnected_nodes'], NotImplementedError('The given task hasn\'t implemented. Please double check')
        self.task_select = True

        basic_texts = {
            'zero_shot': "You are an expert on graph network analysis, tasked with solving the graph problem based on the provided graph data. Please respond concisely and directly to the user\'s queries.",
            'few_shot': "You are an expert on graph network analysis, tasked with solving the graph problem based on the provided graph data. First, you'll be shown an example to familiarize yourself with the response format. Then, you\'ll need to answer using the same format.",
            'cot': "You are an expert on graph network analysis, tasked with solving the graph problem based on the provided graph data. First, you'll be shown an example to familiarize yourself with the response format. Then, you\'ll need to answer using the same format.",
            'cg': "You are tasked with assisting a user who is seeking help with graph networks and Python programming. The user is looking for guidance from an AI that is knowledgeable in graph networks, proficient in Python, and capable of providing bug-free solutions to graph network problems. You will first been given an example to learn about the way to answer. Then you should adhere strictly to the provided program in the example to answer the graph network question."
        }

        formatted_constrain_texts = {
            'zero_shot': {
                'edge_existence': "Your final answer should be 'Yes' or 'No' to whether there is an edge between the node i and the node j, in the form '\\boxed{answer}', at the end of your response. If the graph does not mention the connection between node i and node j, your response should be '\\boxed{No}'.",
                'node_degree': "Your final answer should be a single numerical number, in the form '\\boxed{answer}', at the end of your response. If no information about the node is available or if there are no nodes, your answer should be '\\boxed{0}'.",
                'node_count': "Your final answer should be a single numerical number, in the form '\\boxed{answer}', at the end of your response. If no information about the node is available or if there are no nodes, your answer should be '\\boxed{0}'.",
                'edge_count': "Your final answer should be a single numerical number, in the form '\\boxed{answer}', at the end of your response. If no information about the edge is available or if there are no edges, your answer should be '\\boxed{0}'.",
                'cycle_check': "Your final answer should be 'Yes' or 'No' to whether there is cycle in the undirected graph G, in the form '\\boxed{answer}', at the end of your response. If there is not enough information to answer the question, your response should be '\\boxed{No}'.",
                'connected_nodes': "Your final answer should list all nodes connected to the specified node in alphabetical order, separated by commas, and excluding the specified node itself. For example, if connected to node D are nodes A, C, and B, your response should be '\\boxed{A, B, C}', at the end of your response.  If no nodes are connected to the specified node, your response should be '\\boxed{No nodes}'."
            },
            'few_shot': {
                'edge_existence': "If there is not enough information to answer the question, your final answer should be 'No', in the form '\\boxed{No}', at the end of your response.",
                'node_degree': "If there is not enough information to answer the question, your final answer should be '0', in the form '\\boxed{0}', at the end of your response.",
                'node_count': "If there is not enough information to answer the question, your final answer should be '0', in the form '\\boxed{0}', at the end of your response.",
                'edge_count': "If there is not enough information to answer the question, your final answer should be '0', in the form '\\boxed{0}', at the end of your response.",
                'cycle_check': "If there is not enough information to answer the question, your final answer should be 'No', in the form '\\boxed{No}', at the end of your response.",
                'connected_nodes': "If there is not enough information to answer the question, your final answer should be 'No nodes', in the form '\\boxed{No nodes}', at the end of your response."
            },
            'cot': {
                'edge_existence': "If there is not enough information to answer the question, your final answer should be 'No', in the form '\\boxed{No}', at the end of your response.",
                'node_degree': "If there is not enough information to answer the question, your final answer should be '0', in the form '\\boxed{0}', at the end of your response.",
                'node_count': "If there is not enough information to answer the question, your final answer should be '0', in the form '\\boxed{0}', at the end of your response.",
                'edge_count': "If there is not enough information to answer the question, your final answer should be '0', in the form '\\boxed{0}', at the end of your response.",
                'cycle_check': "If there is not enough information to answer the question, your final answer should be 'No', in the form '\\boxed{No}', at the end of your response.",
                'connected_nodes': "If there is not enough information to answer the question, your final answer should be 'No nodes', in the form '\\boxed{No nodes}', at the end of your response."
            },
        }

        task_texts = {
            'edge_existence': {
                'adjacency': "For this task, please determine whether there is an an edge between node i and node j in the undirected graph G.",
                'coauthorship': "For this task, please determine whether there is an edge between node i and node j in the undirected graph G.",
                'incident': "For this task, please determine whether there is an edge between node i and node j in the undirected graph G.",
                'expert': "For this task, please determine whether there is an edge between node i and node j in the undirected graph G.",
                'friendship': "For this task, please determine whether there is an edge between node i and node j in the undirected graph G.",
                'social_network': "For this task, please determine whether there is an edge between node i and node j in the undirected graph G.",
                'got': "For this task, please determine whether there is an edge between node i and node j in the undirected graph G."
            },
            'node_degree': {
                'adjacency': "For this task, please count the degree of a node in the undirected graph G.",
                'coauthorship': "For this task, please count the degree of a node in the undirected graph G.",
                'incident': "For this task, please count the degree of a node in the undirected graph G.",
                'expert': "For this task, please count the degree of a node in the undirected graph G.",
                'friendship': "For this task, please count the degree of a node in the undirected graph G.",
                'social_network': "For this task, please count the degree of a node in the undirected graph G.",
                'got': "For this task, please count the degree of a node in the undirected graph G."
            },
            'node_count': {
                'adjacency': "For this task, please count the number of nodes in the undirected graph G.",
                'coauthorship': "For this task, please count the number of nodes in the undirected graph G.",
                'incident': "For this task, please count the number of nodes in the undirected graph G.",
                'expert': "For this task, please count the number of nodes in the undirected graph G.",
                'friendship': "For this task, please count the number of nodes in the undirected graph G.",
                'social_network': "For this task, please count the number of nodes in the undirected graph G.",
                'got': "For this task, please count the number of nodes in the undirected graph G."
            },
            'edge_count': {
                'adjacency': "For this task, you will count the number of edges in the undirected graph G.",
                'coauthorship': "For this task, you will count the number of edges in the undirected graph G.",
                'incident': "For this task, you will count the number of edges in the undirected graph G.",
                'expert': "For this task, you will count the number of edges in the undirected graph G.",
                'friendship': "For this task, you will count the number of edges in the undirected graph G.",
                'social_network': "For this task, you will count the number of edges in the undirected graph G.",
                'got': "For this task, you will count the number of edges in the undirected graph G."
            },
            'cycle_check': {
                'adjacency': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, (i,j) means that node i and node j are connected with an undirected edge.",
                'coauthorship': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, i and j wrote a paper together means that i and j are connected with an undirected edge.",
                'incident': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, node i is connected to node j means that they are connected with an undirected edge.",
                'expert': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, i -> j means that node i and node j are connected with an undirected edge.",
                'friendship': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, i and j are friends means that i and j are connected with an undirected edge.",
                'social_network': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, i and j are connected means that i and j are connected with an undirected edge.",
                'politician': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, i and j are connected means that i and j are connected with an undirected edge.",
                'got': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, i and j are friends means that i and j are connected with an undirected edge.",
                'south_park': "For this task, please determine if the given undirected graph G contains any cycles. In the undirected graph, i and j are friends means that i and j are connected with an undirected edge."
            },

            'connected_nodes': {
                'adjacency': "For this task, please list all nodes that are connected to the specified node in the undirected graph G.",
                'coauthorship': "For this task, please list all nodes that are connected to the specified node in the undirected graph G.",
                'incident': "For this task, please list all nodes that are connected to the specified node in the undirected graph G.",
                'expert': "For this task, please list all nodes that are connected to the specified node in the undirected graph G.",
                'friendship': "For this task, please list all nodes that are connected to the specified node in the undirected graph G.",
                'social_network': "For this task, please list all nodes that are connected to the specified node in the undirected graph G.",
                'south_park': "For this task, please list all nodes that are connected to the specified node in the undirected graph G."
            }
        }

        self.basic_text = basic_texts[self.prompt_method].strip()
        if not self.prompt_method == 'cg':
            self.formatted_constrain_text = formatted_constrain_texts[self.prompt_method][task].strip()
        self.task_specific_message_text = task_texts[task][text_enc].strip()


    def get_token_usage(self,response):
        if self.model == 'GPT35':
            return response.usage.total_tokens
        else:
            return response.usage.completion_tokens
        
   
    def data_input(self, question: str):
        prompt = (
        f"{self.basic_text}\n"
        f"{self.task_specific_message_text}\n"
        f"{question}\n\n"
        f"{self.formatted_constrain_text}"
        ).strip()
        assert type(prompt) is str, "prompt must be a string"
        self.message_text.append(
            {'role': 'user',
             'content': prompt}
        )
        if self.model == 'GPT35':
            response = self.client.chat.completions.create(
            model=self.model,
            messages=self.message_text,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
            )
        else:
            response = self.client.chat.completions.create(
            model=self.model,
            messages=self.message_text,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            max_tokens= self.max_token
            )

        self.message_text.pop()
        return response.choices[0].message.content, self.get_token_usage(response)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, choices=['edge_existence', 'node_degree', 'node_count', 'edge_count', 'cycle_check'])
    parser.add_argument('--text_enc', type=str, required=True, choices=['adjacency', 'coauthorship', 'incident', 'expert'])
    parser.add_argument('--prompt_method', type=str, required=True, choices=['zero_shot', 'few_shot', 'cot', 'cg'])
    args = parser.parse_args()

