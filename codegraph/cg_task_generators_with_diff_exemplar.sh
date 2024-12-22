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
# Include two input arguments (`QUESTION_ALGORITHM` and `EXEMPLAR_ALGORITHM`) to enable flexible combinations of algorithms for question and exemplar settings.

#!/bin/bash
set -e
set -x
# Default k-shot is 1 unless specified
K_SHOT=${3:-1}
# Algorithms to process, passed as arguments
QUESTION_ALGORITHM=$1
EXEMPLAR_ALGORITHM=$2

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegraph

GRAPHS_DIR="./graphqa/graphs"
BASE_TASK_DIR="./codegraph/tasks"
TASK_DIR="${BASE_TASK_DIR}/${QUESTION_ALGORITHM}_${EXEMPLAR_ALGORITHM}"
mkdir -p "${TASK_DIR}" # Ensure the directory exists

TASKS=("node_degree" "connected_nodes")

# Generating examples for a specific algorithm
echo "Generating tasks using question algorithm $QUESTION_ALGORITHM and exemplar algorithm $EXEMPLAR_ALGORITHM"
for TASK in "${TASKS[@]}"
do
    echo "Generating prompts for task $TASK"
    python3 -m codegraph.cg_graph_task_generator_with_diff_exemplar \
                --task=$TASK \
                --question_algorithm=$QUESTION_ALGORITHM \
                --exemplar_algorithm=$EXEMPLAR_ALGORITHM \
                --task_dir=$TASK_DIR \
                --graphs_dir=$GRAPHS_DIR \
                --random_seed=1234 \
                --k_shot=$K_SHOT
done
