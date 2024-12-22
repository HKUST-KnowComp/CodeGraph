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


set -e
set -x

mkdir -p logs


# Define the algorithms to be processed
QUESTION_ALGORITHMS=("ba" "sbm" "sfn" "complete" "star" "path")
EXEMPLAR_ALGORITHMS=("er")

# Loop over the pairs of algorithms and create a background process for each pair
for QUESTION_ALGORITHM in "${QUESTION_ALGORITHMS[@]}"
do
    for EXEMPLAR_ALGORITHM in "${EXEMPLAR_ALGORITHMS[@]}"
    do
        SESSION_NAME="generate_${QUESTION_ALGORITHM}_${EXEMPLAR_ALGORITHM}"
        
        # Run each task in the background and log the output to separate files
        bash ./codegraph/cg_task_generators_with_diff_exemplar.sh "$QUESTION_ALGORITHM" "$EXEMPLAR_ALGORITHM" "$K_SHOT" > "logs/${SESSION_NAME}.log" 2>&1 &
        
        # Capture the process ID to track running processes (for debugging)
        echo "Launched session ${SESSION_NAME} with PID: $!"
    done
done

wait

echo "All sessions have been launched. Logs are available in the 'logs' folder."