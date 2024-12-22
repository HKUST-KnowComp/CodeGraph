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
# - Dynamically created task directories for each algorithm.
# - Support k-shot

#!/bin/bash
set -e
set -x
# Default k-shot is 1 unless specified
K_SHOT=${1:-1}

# Now you have $K_SHOT available in this script
echo "Generating tasks for ${K_SHOT}-shot..."

# python3 -m venv graphqa
# source graphqa/bin/activate

# Fill in appropriate output path
GRAPHS_DIR="./graphqa/graphs"
BASE_TASK_DIR="./codegraph/tasks"
TASKS=("edge_existence" "node_degree" "node_count" "edge_count" "cycle_check" "connected_nodes")
ALGORITHMS=('er' 'sbm' 'sfn' 'complete' 'star' 'path' 'ba')

for algorithm in "${ALGORITHMS[@]}"
do
  # Set the task directory for the current algorithm
  TASK_DIR="${BASE_TASK_DIR}/${algorithm}"
  mkdir -p "$TASK_DIR"

  echo "The output path for algorithm $algorithm is set to: $TASK_DIR"

  for task in "${TASKS[@]}"
  do
    echo "Generating examples for task $task using algorithm $algorithm"
    python3 -m codegraph.cg_graph_task_generator \
                --task=$task \
                --algorithm=$algorithm \
                --task_dir=$TASK_DIR \
                --graphs_dir=$GRAPHS_DIR \
                --random_seed=1234 \
                --k_shot=$K_SHOT
  done
done
