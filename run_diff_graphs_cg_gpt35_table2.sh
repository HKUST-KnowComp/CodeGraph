# # coding=utf-8
# #Copyright 2024 HKUST-KnowComp Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegraph

# Define arrays for tasks, encodings, and other configurations
TASK_NAMES=("edge_count" "connected_nodes" "cycle_check" "node_count" "node_degree" "edge_existence")
TEXT_ENCS=("adjacency" "friendship")
GRAPH_GENS=("er" "ba" "sbm" "sfn" "complete" "star" "path")
PROMPT_METHOD="cg"
MODEL_NAMES=("GPT35")
PROMPT_SOURCE="codegraph"
NUMBER_OF_QUESTIONS=500  # Default to 1 if not provided as a command-line argument
K_SHOT=1

# Option to enable or disable logging
ENABLE_LOGGING=true

MAX_PARALLEL_JOBS=16
COUNTER=0
NUM_HEADER_LINES=2

declare -a TASK_LIST
declare -A TASK_STATUS
declare -A TASK_INDEX_MAP

TASK_INDEX=0

print_header() {
    printf "%-5s %-10s %-10s %-15s %-15s %-15s %-15s %-10s\n" \
        "ID" "MODEL" "PROMPT" "TASK" "ENCODING" "GRAPH_GEN" "STATUS" "PID"
    printf '%*s\n' 110 | tr ' ' '-'
}

print_initial_table() {
    print_header
    for ((i=0; i<TOTAL_TASKS; i++)); do
        IFS=';' read -r -a INFO <<< "${TASK_LIST[$i]}"
        TID=${INFO[0]}
        MODEL=${INFO[1]}
        PMETHOD=${INFO[2]}
        TNAME=${INFO[3]}
        TENC=${INFO[4]}
        GGEN=${INFO[5]}
        printf "%-5s %-10s %-10s %-15s %-15s %-15s %-15s %-10s\n" \
            $(printf "%04d" "$TID") "$MODEL" "$PMETHOD" "$TNAME" "$TENC" "$GGEN" "Pending" "--"
    done
}

update_task_status() {
    local TASK_ID=$1
    local STATUS=$2
    local PID=$3

    local TASK_INDEX=${TASK_INDEX_MAP[$TASK_ID]}
    TASK_STATUS[$TASK_ID]=$STATUS
    IFS=';' read -r -a TASK_INFO <<< "${TASK_LIST[$TASK_INDEX]}"
    TASK_INFO[6]=$PID
    TASK_LIST[$TASK_INDEX]=$(IFS=';'; echo "${TASK_INFO[*]}")

    local LINE_NUM=$((TASK_INDEX + NUM_HEADER_LINES))
    tput cup $LINE_NUM 0
    tput el
    printf "%-5s %-10s %-10s %-15s %-15s %-15s %-15s %-10s\n" \
        "$(printf "%04d" ${TASK_INFO[0]})" "${TASK_INFO[1]}" "${TASK_INFO[2]}" \
        "${TASK_INFO[3]}" "${TASK_INFO[4]}" "${TASK_INFO[5]}" \
        "${STATUS}" "${PID}"
}

declare -a JOB_PIDS
declare -a JOB_TASK_IDS

cleanup() {
    echo "Stopping all running jobs..."
    for pid in "${JOB_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "Stopped process $pid"
        fi
    done
    exit 1
}

trap cleanup EXIT SIGINT

wait_for_one_job() {
    local PID=${JOB_PIDS[0]}
    local TASK_ID=${JOB_TASK_IDS[0]}
    wait $PID
    if [ $? -eq 0 ]; then
        update_task_status "$TASK_ID" "Completed" "--"
    else
        update_task_status "$TASK_ID" "Failed" "--"
    fi
    COUNTER=$((COUNTER - 1))
    JOB_PIDS=("${JOB_PIDS[@]:1}")
    JOB_TASK_IDS=("${JOB_TASK_IDS[@]:1}")
}

# Build the task list
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for GRAPH_GEN in "${GRAPH_GENS[@]}"; do
        for TEXT_ENC in "${TEXT_ENCS[@]}"; do
            for TASK_NAME in "${TASK_NAMES[@]}"; do
                TASK_ID="$TASK_INDEX"
                TASK_STATUS[$TASK_ID]="Pending"
                TASK_LIST[$TASK_INDEX]="$TASK_ID;$MODEL_NAME;$PROMPT_METHOD;$TASK_NAME;$TEXT_ENC;$GRAPH_GEN;--"
                TASK_INDEX_MAP[$TASK_ID]=$TASK_INDEX
                TASK_INDEX=$((TASK_INDEX + 1))
            done
        done
    done
done

TOTAL_TASKS=${#TASK_LIST[@]}

clear
print_initial_table

# Ensure the table is fully printed before starting tasks
# Flush output and give a brief pause
sleep 1.0

# Now start tasks AFTER the entire table has been printed
for ((i=0; i<TOTAL_TASKS; i++)); do
    IFS=';' read -r -a TASK_INFO <<< "${TASK_LIST[$i]}"
    TASK_ID=${TASK_INFO[0]}
    MODEL_NAME=${TASK_INFO[1]}
    PMETHOD=${TASK_INFO[2]}
    TASK_NAME=${TASK_INFO[3]}
    TEXT_ENC=${TASK_INFO[4]}
    GRAPH_GEN=${TASK_INFO[5]}

    LOG_DIR="logs/$PROMPT_SOURCE/$MODEL_NAME/${PMETHOD}/$GRAPH_GEN/$TASK_NAME"
    if [ "$ENABLE_LOGGING" = true ]; then
        mkdir -p "$LOG_DIR" || { echo "Error: Failed to create log directory $LOG_DIR"; exit 1; }
        LOG_FILE="$LOG_DIR/${TEXT_ENC}.log"
    fi

    if [ "$ENABLE_LOGGING" = true ]; then
        python evaluate.py \
            --prompt_source "$PROMPT_SOURCE" \
            --prompt_method "$PMETHOD" \
            --task_name "$TASK_NAME" \
            --text_enc "$TEXT_ENC" \
            --graph_gen "$GRAPH_GEN" \
            --model_name "$MODEL_NAME" \
            --number_of_questions "$NUMBER_OF_QUESTIONS" \
            --k_shot "$K_SHOT" \
            > "$LOG_FILE" 2>&1 &
    else
        python evaluate.py \
            --prompt_source "$PROMPT_SOURCE" \
            --prompt_method "$PMETHOD" \
            --task_name "$TASK_NAME" \
            --text_enc "$TEXT_ENC" \
            --graph_gen "$GRAPH_GEN" \
            --model_name "$MODEL_NAME" \
            --number_of_questions "$NUMBER_OF_QUESTIONS" \
            --k_shot "$K_SHOT" \
            &
    fi

    PID=$!
    update_task_status "$TASK_ID" "Running" "$PID"

    JOB_PIDS+=("$PID")
    JOB_TASK_IDS+=("$TASK_ID")

    COUNTER=$((COUNTER + 1))

    if [ "$COUNTER" -ge "$MAX_PARALLEL_JOBS" ]; then
        wait_for_one_job
    fi
done

while [ "$COUNTER" -gt 0 ]; do
    wait_for_one_job
done

tput cup $((TOTAL_TASKS + NUM_HEADER_LINES + 1)) 0
echo "Evaluations for Table 2 using '$PROMPT_SOURCE' are completed."
