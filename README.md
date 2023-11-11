# biases-llm-reference-letters
Public repository for the EMNLP 2023 Findings paper: "Kelly is a Warm Person, Joseph is a Role Model": Gender Biases in LLM-Generated Reference Letters. Code and data will be released soon!

Arxiv version available at: https://arxiv.org/abs/2310.09219

## Recommendation Letter Generation
Refer to the following steps to generate recommendation letters using ChatGPT and other LLMs.

### Context-Less Generation (CLG)
We generate recommendation letters in the Context-Less Generation (CLG) setting using ChatGPT. To run generation, first add in your OpenAI organization and API key in `generate_clg.py`. Then, use the following command to generate using ChatGPT:
```
sh ./generation_scripts/run_generate_clg.sh
```