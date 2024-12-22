## GraphQA

**a. Clone GraphQA repo.**

To evaluate CodeGraph on GraphQA benchmark, please first clone the [repo of graphqa](https://github.com/google-research/google-research/tree/master/graphqa) and place it under the following folder structure.

**Folder structure**
```
CodeGraph/  
├── codegraph/
├── docs/
├── graphqa/
├── models/
├── README.md
├──run_parallel_cg_graph_generator_with_diff_exemplar.sh
└── ...

```

**b. Generating graphs.**

1. **Navigate to the Project Directory**
   ```shell
   cd CodeGraph
    ```
2. **Modify the Output Path**
   modify the output path in `graph_generator.sh` under the `graphqa` folder
   ```
   OUTPUT_PATH="graphqa/graphs"
   ```
3. **Generate Graphs for Test Set**
   ```bash
   ./graphqa/graph_generator.sh
   ```
4. **Generate Graphs for Train Set**

   **Step 1**

   To generate graphs for the ``train`` split instead of the ``test`` split, modify the ``--split`` in graph_generator.sh:

      ```bash
      --split=train
      ```
      
   **Step 2** 

   ```bash
   ./graphqa/graph_generator.sh
   ```
   