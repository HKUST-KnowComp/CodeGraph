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

"""Evaluate various graph tasks on LLMs."""

import sys
import os
import json
import argparse
from tqdm import tqdm
from time import time

import tensorflow as tf
# Set TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Determine the absolute path of the project root directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Print the project directory to debug
print("PROJECT_DIR:", PROJECT_DIR)
# Add the project root directory to sys.path
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)


from models.clients import Clients
from get_graphqa_answer import (
    extract_connected_nodes,
    extract_cot_num_response,
    extract_num_response,
    extract_yes_no_response,
    exec_py,
    process_answer_to_correct_sequence,
)


def process_ground_truth_answer(answer, task_name):
    """Process the ground truth answer based on the task."""
    if task_name in ['cycle_check', 'edge_existence']:
        if not args.prompt_method == 'cg':
            return extract_yes_no_response(answer)  #this is to convert the ground truth to 'yes' or 'no'
        else:
            return answer
    elif task_name == 'connected_nodes':
        return process_answer_to_correct_sequence(answer)
    else:
        return answer

def extract_model_answer(ans, args, question):
    """Extract the model's answer based on the task and prompt method."""
    if args.task_name in ['node_degree', 'edge_count', 'node_count']:
        if args.prompt_method == 'cot':
            return extract_cot_num_response(ans)
        elif args.prompt_method == 'cg':
            return exec_py(ans)
        else:
            return extract_num_response(ans)
    elif args.task_name in ['cycle_check', 'edge_existence']:
        if args.prompt_method != 'cg':
            return extract_yes_no_response(ans)
        else:
            return exec_py(ans)
    elif args.task_name == 'connected_nodes':
        if args.prompt_method != 'cg':
            return extract_connected_nodes(ans, args.text_enc, question)
        else:
            return str(exec_py(ans))
    else:
        return ans

def log_wrong_case(results, example_id, answer, gpt_answer, response):
    """Log wrong cases."""
    print({
        'ID': example_id,
        'correct_ans': answer,
        'gpt_ans': gpt_answer,
        'response': response
    })
    results['wrong_cases'].append({
        'ID': example_id,
        'correct_ans': answer,
        'gpt_ans': gpt_answer,
        'response': response
    })

def save_results(results, args):
    """Save the results to a JSON file."""
    if args.prompt_method == 'cg':
        save_path = os.path.join(PROJECT_DIR, 'results', args.prompt_source, args.model_name, f"{args.prompt_method}_{args.k_shot}_shot_result", args.graph_gen, args.task_name)
    else:
        save_path = os.path.join(PROJECT_DIR, 'results', args.prompt_source, args.model_name, f"{args.prompt_method}_result", args.graph_gen, args.task_name)
    os.makedirs(save_path, exist_ok=True)
    file_name = f'{args.task_name}_{args.text_enc}.json'
    with open(os.path.join(save_path, file_name), 'w') as f:
        json.dump(results, f, indent=4)

def evaluate(args):
    """Read prompts from TFRecord files and evaluate the performance of the LLMs on the  GraphQA benchmark.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing settings for the evaluation, such as task name, graph type, prompt method, and model.

    Returns:
        dict: A summary of the evaluation results, including:
            - 'Total Count': Total number of questions evaluated.
            - 'Average time used': Average time taken to process each question.
            - 'Average token used': Average number of tokens used by the model per question.
            - 'Total time used': Total time taken for the evaluation.
            - 'Accuracy rate': Accuracy of the model on the graph task.
    """
    """Read prompts from TFRecord files and evaluate the performance of the LLMs on the benchmark."""
    if args.prompt_method == 'cg':
    # For the CodeGraph method, we have a naming pattern that includes k_shot
        dataset_file = f"{args.task_name}_cg_{args.k_shot}_shot_test.tfrecords"
    else:
    # For all other prompting methods (few_shot, zero_shot, cot), stick to the old pattern
        dataset_file = f"{args.task_name}_{args.prompt_method}_test.tfrecords"
    dataset_path = os.path.join(PROJECT_DIR, args.prompt_source, 'tasks', args.graph_gen, dataset_file)
    print(f"Dataset path: {dataset_path}")
    raw_dataset = tf.data.TFRecordDataset(dataset_path)
    feature_description = {
        'question': tf.io.FixedLenFeature([], tf.string),
        'answer': tf.io.FixedLenFeature([], tf.string),
        'nnodes': tf.io.FixedLenFeature([], tf.string),
        'nedges': tf.io.FixedLenFeature([], tf.string),
        'algorithm': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.string),
        'text_encoding': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        """Parse a single TFRecord example."""
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    results = {
        'summary': {
            'Total Count': 0,
            'Average time used': 0.0,
            'Average token used': 0.0,
            'Total time used': 0.0,
            'Accuracy rate': 0.0,
            'Model Name': args.model_name,
            'Graph Generator Algorithm': args.graph_gen,
            'Prompting Method': args.prompt_method,
        },
        'wrong_cases': [],
    } 

    correct_count = 0
    total_count = 0
    total_time = 0.0
    total_token = 0
    total_examples = args.number_of_questions
    for example in tqdm(parsed_dataset, total=total_examples):
        start_time = time()
        text_encoder = example['text_encoding'].numpy().decode('utf-8')
        if text_encoder != args.text_enc:
            continue
        question = example['question'].numpy().decode('utf-8')
        try:
            # Process the ground truth answer
            answer_raw = example['answer'].numpy().decode('utf-8')
            try:
                answer = int(answer_raw.rstrip('.'))  # Remove trailing period if present
            except ValueError:
                answer = answer_raw
            answer = process_ground_truth_answer(answer, args.task_name)
            # Send question to the model
            ans, token_count = graph_gpt.data_input(question)
        except Exception as e:
            # Log error and continue
            example_id = example['id'].numpy().decode('utf-8')
            log_wrong_case(results, example_id, answer, str(e), 'NA')
            total_count += 1
            total_time += time() - start_time
            continue
        total_token += token_count
        # Extract model's answer
        gpt_answer = extract_model_answer(ans, args, question)
        # Compare answers
        if gpt_answer == answer:
            correct_count += 1
        else:
            example_id = example['id'].numpy().decode('utf-8')
            log_wrong_case(results, example_id, answer, gpt_answer, ans)
        total_count += 1
        total_time += time() - start_time
        if args.debug and total_count == 10:
            break
        if args.number_of_questions == total_count:
            break
    # Update results summary
    results['summary']['Total Count'] = total_count
    results['summary']['Average time used'] = total_time / total_count if total_count > 0 else 0
    results['summary']['Average token used'] = total_token / total_count if total_count > 0 else 0
    results['summary']['Total time used'] = total_time
    results['summary']['Accuracy rate'] = correct_count / total_count if total_count > 0 else 0
    # Save results
    if not args.debug:
        save_results(results, args)
    print(f'Total Count: {total_count}')
    print(f'Average time used: {results["summary"]["Average time used"]}')
    print(f'Average token used: {results["summary"]["Average token used"]}')
    return results['summary']['Accuracy rate']

if __name__ == "__main__":
    # Ensure the file path is correct
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_source', type=str, default='codegraph', choices=['codegraph', 'graphqa'],
                        help='Specify whether to use the CodeGraph or GraphQA prompts')
    parser.add_argument('--task_name', type=str, required=True,
                        choices=['edge_count', 'connected_nodes', 'cycle_check', 'node_count', 'node_degree', 'edge_existence'])
    parser.add_argument('--text_enc', type=str, required=True, choices=['adjacency', 'coauthorship', 'incident',
                        'expert', 'friendship', 'social_network', 'politician', 'got', 'south_park'])
    parser.add_argument('--graph_gen', type=str, required=True,
                        choices=['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path','path_er','sbm_er','sfn_er','star_er','ba_er','complete_er'])
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--prompt_method', type=str, choices=['few_shot', 'cot', 'zero_shot', 'cg'], required=True, help='Select the prompting method')
    parser.add_argument('--model_name', type=str, default='GPT35', choices=['GPT35', 'Llama_3_70B', 'Mixtral_8x7B', 'Mixtral_8x22B'], help='Specify the model to use for querying')
    parser.add_argument('--number_of_questions', type=int, default=500, help='Number of questions to evaluate')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of shots to use for the codegraph prompting (default: 1)')
    args = parser.parse_args()
    program_start_time = time()
    graph_gpt = Clients(model_name=args.model_name)
    graph_gpt.prompt_selection(prompt_method=args.prompt_method)
    graph_gpt.task_selection(task=args.task_name, text_enc=args.text_enc)
    acc_rate = evaluate(args)
    print(f'Time used: {time() - program_start_time}')
    print(f'Accuracy rate: {acc_rate}')