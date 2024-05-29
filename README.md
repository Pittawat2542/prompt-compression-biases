# Assessing Inherent Biases Following Prompt Compression of Large Language Models for Game Story Generation

This repository contains the code and datasets for the paper "Assessing Inherent Biases Following Prompt Compression of Large Language Models for Game Story Generation" accepted at [IEEE CoG 2024](https://2024.ieee-cog.org).

## Authors
Pittawat Taveekitworachai, Kantinan Plupattanakit, Ruck Thawonmas

## Abstract

This paper investigates how prompt compression, a technique to reduce the number of tokens in the prompt while maintaining prompt performance, affects inherent biases in large language models (LLMs) for the story ending of the game story generation task. Previous studies have explored inherent biases in LLMs and found an innate inclination of LLMs towards generating positive-ending stories. While prompt compression is known to retain task performance and utilize fewer tokens in the prompt, we explore a different perspective on how prompt compression could affect inherent biases in LLMs. We follow existing studies' approach in evaluating story ending biases of six LLMs comparing uncompressed and compressed prompts. We find that prompt compression does not affect story generation from positive-ending story synopses, to which these LLMs are inclined. However, it is not the same for negative-ending story synopses: prompt compression either makes the LLMs generate a higher amount of negative-ending stories or not at all. We also notice that the classification of other types of story endings, other than those specified in the prompt, aligns with an existing study. We recommend game developers and future studies to always perform empirical tests on prompt compression, as it is not straightforward and may greatly alter model behaviors.

## File structure
```
.
├── analysis.ipynb # This file contains the analysis of the results
├── main.py # This file contains the main code to run the experiments
├── outputs.zip # This file contains the outputs of the experiments
├── requirements.txt # This file contains the dependencies of the project
├── src # This folder contains the source code of the project
│   ├── __init__.py
│   ├── llms # This folder contains the code of the LLMs
│   ├── models # This folder contains the code of the data models
│   └── prompts.py # This file contains the prompts of the experiments
├── stat-analysis.ipynb # This file contains the statistical analysis of the results
└── utils # This folder contains the utility code of the project
    ├── __init__.py
    ├── llms.py # This file contains the utility functions of the LLMs
    └── parsers.py # This file contains the utility functions of the parsers
```

## Installation and Usage
0. Create a virtual environment (if needed):
```bash
conda create -n prompt-compress python=3.12
```
and activate it:
```bash
conda activate prompt-compress
```
1. Copy `.env.example` and rename it to `.env`. Follow instructions on [this page](https://platform.openai.com/docs/api-reference/authentication) to obtain your own OpenAI API key. Add API keys of other LLMs as needed.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Run the code.
```bash
python main.py [command]
```

### Available commands
1. `generate-story-data`: This command is used to generate the data for the story generation task.
2. `generate-story-influence`: This command is used to generate story influenced by the data generated in the previous step. It should be run after the `generate-story-data` command.
3. `evaluate-story`: This command is used to evaluate the generated stories. It should be run after the `generate-story-influence` command. 
4. `analyse`: This commmand is used to analyse the results of the prompt compression. It should be run after the `evaluate-story` command.

For additional information on the commands, run:
```bash
python main.py [command] --help
```
