## Installation Instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n codegraph python=3.8
conda activate codegraph
```
**b. Install reuired pacakges following the [GraphQA requirements](https://github.com/google-research/google-research/blob/bfa1a6eaaac2bbde8ab6a376de6974233b7456c1/graphqa/requirements.txt).**

```shell
pip install tensorflow absl-py networkx numpy tqdm openai
```

**c. Clone CodeGraph.**
```
git clone https://github.com/HKUST-KnowComp/CodeGraph.git
```

## Set Up API Credentials

Set Environment Variables:
- For GPT-3.5 model:
```shell
export AZURE_API_KEY='your_azure_api_key'
export AZURE_ENDPOINT='your_azure_endpoint'
```
- For Other Models (e.g, Llama, Mistral)
```shell
export DEEPINFRA_API_KEY='your_deepinfra_api_key'
export DEEPINFRA_BASE_URL='https://api.deepinfra.com/v1/openai'
```
