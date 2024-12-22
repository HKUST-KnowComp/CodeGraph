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

import sys
import os
import subprocess
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
import re
from graphqa import name_dictionaries
from graphqa.graph_text_encoder import TEXT_ENCODER_DICT

"""Code to extract answers"""

def process_answer_to_correct_sequence(answer):
    answer = str(answer).strip().rstrip('.')
    elements = answer.split(',')
    if all(element.strip().isdigit() for element in elements):
        sorted_elements = sorted(int(element) for element in elements)
        return ', '.join(map(str, sorted_elements))
    else:
        return answer


def exec_py(code:str):
    try:
        # Using a regular expression to flexibly match the start and end delimiters
        pattern = r'(?i)#\s*CODE\s+START\n(.*?)#\s*CODE\s+END'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            new_code = match.group(1).strip() + "\nprint(ans)"
            #import pdb; pdb.set_trace()
        else:
            return "Code snippet not found."

        # Execute the extracted code
        result = subprocess.run(['python', '-c', new_code], capture_output=True, text=True, check=True)
        resp = result.stdout.strip()

        ans_list = [line.strip() for line in resp.split('\n') if line.strip()]
        ans = ans_list[-1] if ans_list else None

        # Try converting to integer if possible
        try:
            ans = int(ans)
        except ValueError:
            ans = ans

        return ans
    except Exception as e:
        print(f'exception {e} in exec_py() with input text: {code}')
        return -1

def extract_num_response(resp:str):
    try:
        # First try to extract numbers from the expected LaTeX boxed format
        pattern_boxed = r'\\boxed\{(\d+)\}'
        numbers = re.findall(pattern_boxed, resp)
        if numbers:
            return int(numbers[0])
        
        # If no number is found in the boxed format, try to extract any number at the end of the response
        pattern_general = r'\b(\d+)\b(?![\s\S]*\b\d+\b)'
        numbers_general = re.findall(pattern_general, resp)
        if numbers_general:
            return int(numbers_general[0])
        
        # If no numbers are found at all, return -1
        return -1
    except Exception as e:
        print(f'exception {e} in extract_num_response() with input text: {resp}')
        return -1

def extract_yes_no_response(resp:str):
    try:
        if "Yes" in resp:
            return "Yes"
        elif "No" in resp:
            return "No"
    except Exception as e:
        print(f'exception {e} in extract_num_response() with input text: {resp}')
        return -1

def extract_cot_num_response(resp: str):
    try:
        # First, try to extract numbers immediately following "A:"
        pattern_a_followed = r'A:\s*(\d+)'
        numbers_a = re.findall(pattern_a_followed, resp)
        if numbers_a:
            return int(numbers_a[0])

        # Next, try to extract numbers from the LaTeX boxed format
        pattern_boxed = r'\\boxed\{(\d+)\}'
        numbers_boxed = re.findall(pattern_boxed, resp)
        if numbers_boxed:
            return int(numbers_boxed[0])

        # If no number is found with the above methods, try to extract any number at the beginning of the response
        pattern_general = r'^\d+'
        numbers_general = re.findall(pattern_general, resp)
        if numbers_general:
            return int(numbers_general[0])

        # If no numbers are found at all, return -1
        return -1
    except Exception as e:
        print(f'exception {e} in extract_num_response() with input text: {resp}')
        return -1

def extract_node_in_question(question):
    """
    Extract the node in question from a given string using regular expressions.
    
    Parameters:
    question (str): The question text from which to extract the node in question.
    
    Returns:
    str: The extracted node name or None if not found.
    """
    # Pattern attempts to capture various phrasings of the question
    # This pattern assumes that the node's name or identifier is mentioned right before "in alphabetical order"
    pattern = r"List all the nodes connected to ([\w\s]+) in alphabetical order"
    matches = re.findall(pattern, question)
    if matches:
        target_node = matches[-1].strip()
        try:
            target_node=int(target_node)
        except:
            target_node = target_node
        return target_node
    return None


def extract_connected_nodes(response, encoding_method,question):
    """
    Extract nodes from ChatGPT's response and compare with ground truth, both in string format.

    Parameters:
    response (str): The response from ChatGPT
    ground_truth_str (str): A comma-separated string of nodes that are the ground truth connected nodes, possibly ending with "No nodes."

    Returns:
    str: A string representation of the nodes extracted from the response, formatted as the ground truth.
    """
    # Normalize the ground truth string by removing any trailing period and spaces
    node_name_dict = TEXT_ENCODER_DICT[encoding_method]
    is_numeric = all(item.isdigit() for item in node_name_dict.values())
    #ground_truth_str = ground_truth_str.strip()
    if "No nodes" in response:
        return " No nodes".strip()
    return extract_bboxed_response_and_normal_response(response, is_numeric, node_name_dict,question)

def extract_bboxed_response_and_normal_response(response, is_numeric, node_name_dict,question):
        # Attempt to extract from boxed format
        start = response.find('\\boxed{')
        end = response.find('}', start)
        if start != -1 and end != -1:
            extracted_nodes = response[start + 7:end].strip().replace(' ', '').split(',')
            extracted_nodes = [node.rstrip('.') for node in extracted_nodes] 
            nodes_list = [int(node) for node in extracted_nodes if node.isdigit()] if is_numeric else extracted_nodes
        else:
            # If no boxed format, clean and split the response
            response_nodes = response.replace(',', ' ').replace('.', ' ').split()
            #import pdb ; pdb.set_trace()
            nodes_list = [int(node) for node in response_nodes if node.isdigit() and node in node_name_dict.values()] if is_numeric else [node for node in response_nodes if node in node_name_dict.values()]
        

        # Intersect and sort final list as per the type of ground truth
        final_nodes_list = sorted(set(nodes_list) & set(int(node) for node in node_name_dict.values()), key=int) if is_numeric else sorted(set(nodes_list) & set(node_name_dict.values()))
        
        # Remove the target node if it's in the list
        target_node_in_question = extract_node_in_question(question)
        if target_node_in_question in final_nodes_list:
            final_nodes_list.remove(target_node_in_question)

        return ', '.join(map(str, final_nodes_list)).strip()




if __name__ == '__main__':
    prompt = """
# CODE START
edges = [('James', 'John'),
('John', 'Jennifer'),
('Mary', 'Jennifer'),
('Mary', 'William'),
('Linda', 'Elizabeth'),
('Elizabeth', 'Barbara'),
('William', 'Barbara'),
('Joseph', 'Christopher')
]
def count_nodes(edges):
    nodes = set()
    for node1, node2 in edges:
        nodes.add(node1)
        nodes.add(node2)
    return len(nodes)
answer = count_nodes(edges)
# CODE END
print(ans)
""".strip()
    print(exec_py(prompt))
