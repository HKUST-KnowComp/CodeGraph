# Preparing Prompts for Experiments
## Table of Contents

1. [Graph Tasks Implemented](#graph-tasks-implemented)
2. [Generating Files for Different Graph Tasks](#generating-files-for-different-graph-tasks)
    - [CodeGraph](#codegraph)
        - [Setting 1: Same Graph Structure for Exemplars and Test Examples (Default One-Shot)](#setting-1-same-graph-structure-for-exemplars-and-test-examples-default-one-shot)
        - [Setting 2: Different Graph Structures between Exemplars and Test Examples (Default One-Shot)](#setting-2-different-graph-structures-between-exemplars-and-test-examples-default-one-shot)
    - [Baseline Prompting Methods](#baseline-prompting-methods)


## 1. Graph Tasks Implemented

We evaluate the following basic graph tasks in the GraphQA benchmark with various graph generator algorithms and different graph encoding functions.

**Graph Tasks:**

- `node count`
- `edge count`
- `edge existence`
- `node degree`
- `connected nodes`
- `cycle check`

**Graph generator algorithms** include `er`, `sbm`, `sfn`, `complete`, `star`, `path`, and `ba`.

**Graph encoding functions** include `adjacency`, `friendship`, `co-authorship`, `incident`, `social network`, and `expert`.

---

## 2. Generating Files for Different Graph Tasks

In this section, we describe how to generate the necessary files to run the experiments for **CodeGraph** and **Baseline Prompting Methods**.

### CodeGraph

The following command-line instructions are used to generate the files required for running CodeGraph experiments.

#### Setting 1: Same Graph Structure for Exemplars and Test Examples (Default One-Shot)
```bash
cd CodeGraph
```
```bash
./codegraph/cg_task_generator.sh
```
You can specify a different `k-shot` value to include more exemplars by passing the desired number as an argument.

**Example (2-shot):**
```bash
./codegraph/cg_task_generator.sh 2
```

#### Setting 2: Different Graph Structures between Exemplars and Test Examples (Default One-Shot)

```bash
./run_parallel_cg_graph_generator_with_diff_exemplar.sh
```

### Baseline Prompting Methods

The baseline prompting methods (Default 2-shot) are provided by **GraphQA**, which supports various techniques such as:

- **Few-shot**
- **Zero-shot**
- **Chain of Thought (CoT)**

To use these baseline methods, follow these steps:

1. **Create Folders for Graph Tasks by Algorithm**  
   ```bash
   cd CodeGraph
   ```
   ```bash
   for algo in er ba sfn sbm complete star path; do mkdir -p graphqa/tasks/$algo; done
   ```

2. **Edit the task_generator.sh script**  
   Navigate to the `graphqa` folder and modify the `task_generator.sh` script.

3. **Update the directory paths**  
   Ensure the following variables and their order are correctly set before running the script:

   ```bash
   # Algorithm to use. Default is 'er' (Erdos-Rényi graphs).
   # Other options: sbm, sfn, complete, star, path, ba
   ALGORITHM=${ALGORITHM:-"er"}

   # Directory where generated graphs are stored
   GRAPHS_DIR="./graphqa/graphs"

   # Directory where tasks corresponding to the chosen algorithm are stored
   TASK_DIR="./graphqa/tasks/$ALGORITHM"
   ```

4. **Run the Task Generator**  
   Make sure your current working directory is `CodeGraph`. Then, run the task generator script:

   ```bash
   # Optional: Export ALGORITHM if you wish to override the default
   export ALGORITHM="er"  # Using Erdos-Rényi graphs

   # Run the task generator
   ./graphqa/task_generator.sh
   ```

---
