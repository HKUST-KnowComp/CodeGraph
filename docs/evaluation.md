# Evaluation

## Overview

This document provides instructions on how to run the evaluation experiments for our method as presented in our paper. Before proceeding, ensure that you have completed the installation and data preparation steps outlined in [installation](installation.md), [prepare_data](prepare_data.md), and [prepare_prompts](prepare_prompts.md).



## Results

The evaluation scripts are located in the `CodeGraph` directory. Each script corresponds to the evaluations for a specific table in the paper.

### Table 1: Main Results

To reproduce the results in **Table 1**, which evaluates our method (CodeGraph) and baseline for other prompting methods using GPT-3.5, run the following scripts:

#### Our Method (CodeGraph) with GPT-3.5

```bash
cd CodeGraph
./run_cg_gpt35_table1.sh
```

#### Chain-of-Thought Prompting with GPT-3.5

```bash
cd CodeGraph
./run_cot_gpt35_table1.sh
```

#### Few-Shot Prompting with GPT-3.5

```bash
cd CodeGraph
./run_few_shot_gpt35_table1.sh
```

#### Zero-Shot Prompting with GPT-3.5

```bash
cd CodeGraph
./run_zero_shot_gpt35_table1.sh
```

---

### Table 2: Comparison Across Different Graph Generators

To compare the accuracy of various graph generators on different graph tasks using two graph encoding functions with GPT-3.5 Turbo, run:

```bash
cd CodeGraph
./run_diff_graphs_cg_gpt35_table2.sh
```

This script evaluates our method with different graph structures including `"er"`, `"ba"`, `"sbm"`, `"sfn"`, `"complete"`, `"star"`, and `"path"`, using two encoding methods `"adjacency"` and `"friendship"`.

---

### Table 3: Generalization Performance Across Graph Structures

To evaluates the generalization performance of our method across various graph structures using GPT-3.5, run:

```bash
cd CodeGraph
./run_generalization_graphs_cg_gpt35_table3.sh
```

This script evaluates the generalization performance of CodeGraph on tasks such as **Connected Nodes** and **Node Degree** with different graph encoding functions.

---

### Table 4: Evaluations with Different Models

To evaluate our method with different models like `Llama_3_70B`, `Mixtral_8x7B`, and `Mixtral_8x22B`. To do this, modify the `MODEL_NAMES` variable in the existing scripts or create new scripts.

#### Steps to Modify for Table 4:

1. **Copy the Existing Script**

   ```bash
   cd CodeGraph
   cp run_cg_gpt35_table1.sh run_cg_table4.sh
   ```

2. **Edit `run_cg_table4.sh`**

   Open `run_cg_table4.sh` and modify the `MODEL_NAMES` variable:

   ```bash
   MODEL_NAMES=("Llama_3_70B") # Options:  "Mixtral_8x7B", "Mixtral_8x22B"
   ```

3. **Run the Modified Script**

   ```bash
   ./run_cg_table4.sh
   ```

**Note:** Ensure that you have the appropriate API keys and configurations for the new models as per the [installation instructions](installation.md).

---

### Table 5: Evaluations with K-Shot Settings

To evaluate our method with different numbers of shots (e.g., 2-shots, 3-shots). Ensure that you have prepare the prompts before evaluation.

#### Steps to Modify for Table 5:

1. **Copy the Existing Script**

   ```bash
   cd CodeGraph
   cp run_cg_gpt35_table1.sh run_cg_shots_table5.sh
   ```

2. **Edit `run_cg_shots_table5.sh`**

   Modify the `K_SHOT` variable according to the prepared prompt:

   **e.g:**

   ```bash
   K_SHOT=2 
   ```
   **Note:** The `K_SHOT` value should match the number of exemplars used in the prepared prompts. For instance, if you created 2-shot prompts via `prepare_prompts.md`, set `K_SHOT=2` in the script.

3. **Run the Modified Script**

   ```bash
   ./run_cg_shots_table5.sh
   ```


## Customizing Evaluations

You can customize the evaluation scripts by modifying variables to evaluate different tasks, encodings, graph generators, prompt methods, models, number of questions, and number of shots.

### Variables to Modify

- **Task Names** (`TASK_NAMES`):

  ```bash
  TASK_NAMES=("edge_count" "connected_nodes" "cycle_check" "node_count" "node_degree" "edge_existence")
  ```

- **Text Encodings** (`TEXT_ENCS`):

  ```bash
  TEXT_ENCS=("adjacency" "coauthorship" "incident" "expert" "friendship" "social_network")
  ```

- **Graph Generators** (`GRAPH_GENS`):

  ```bash
  GRAPH_GENS=("er" "ba" "sbm" "sfn" "complete" "star" "path")
  ```

- **Prompt Method** (`PROMPT_METHOD`):

  ```bash
  PROMPT_METHOD="cg"  # Options: "cg", "few_shot", "cot", "zero_shot"
  ```

- **Model Names** (`MODEL_NAMES`):

  ```bash
  MODEL_NAMES=("GPT35" "Llama_3_70B" "Mixtral_8x7B" "Mixtral_8x22B")
  ```

- **Number of Questions** (`NUMBER_OF_QUESTIONS`):

  ```bash
  NUMBER_OF_QUESTIONS=500
  ```
- **Number of Shots** (`K_SHOTS`):

  ```bash
  K_SHOTS=1  # Specify the number of exemplars for the cg method
  ```


## Running Evaluations Manually

You can also run evaluations manually using the `evaluate.py` script.

### Script Arguments

- **`--prompt_source`**: Specify the prompt source (`codegraph` or `graphqa`).
- **`--task_name`**: Choose from `edge_count`, `connected_nodes`, `cycle_check`, `node_count`, `node_degree`, `edge_existence`.
- **`--text_enc`**: Choose from `adjacency`, `coauthorship`, `incident`, `expert`, `friendship`, `social_network`, `politician`, `got`, `south_park`.
- **`--graph_gen`**: Choose from `er`, `ba`, `sbm`, `sfn`, `complete`, `star`, `path`, and their variants.
- **`--prompt_method`**: Choose from `few_shot`, `cot`, `zero_shot`, `cg`.
- **`--model_name`**: Choose from `GPT35`, `Llama_3_70B`, `Mixtral_8x7B`, `Mixtral_8x22B`.
- **`--number_of_questions`**: Specify the number of questions to evaluate.
- **`--k_shot`**: Specify the number of exemplars used for the codegraph method.

### Example Command

```bash
python evaluate.py \
    --prompt_source "codegraph" \
    --task_name "node_count" \
    --text_enc "adjacency" \
    --graph_gen "er" \
    --prompt_method "cg" \
    --model_name "Llama_3_70B" \
    --number_of_questions 500 \
    --k_shot 1
```



## Additional Information

- **Logs and Results**: The records for each thread and the evaluation results will be stored in the `logs` and `results` directories, respectively.
- **API Keys**: Ensure you have set up your API credentials as per the [installation instructions](installation.md).
- **Contact**: For questions or issues, please refer to the [README.md](../README.md) or contact the project maintainers.

---

[Back to Top](#evaluation)
```