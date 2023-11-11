# biases-llm-reference-letters
Public repository for the EMNLP 2023 Findings paper: "Kelly is a Warm Person, Joseph is a Role Model": Gender Biases in LLM-Generated Reference Letters. Code and data will be released soon!

Arxiv version available at: https://arxiv.org/abs/2310.09219

## Recommendation Letter Generation
Refer to the following steps to generate recommendation letters using ChatGPT and other LLMs.

### Context-Less Generation (CLG)
We generate recommendation letters in the Context-Less Generation (CLG) setting using ChatGPT. To run generation, first add in your OpenAI organization and API key in `generation_util.py`. Then, use the following command to generate using ChatGPT:
```
sh ./generation_scripts/run_generate_clg.sh
```
Alternatively, access our generated CLG letters stored in `./generated_letters/chatgpt/clg/clg_letters.csv`

### Context-Based Generation (CBG)
We generate recommendation letters in the Context-Based Generation (CBG) setting using ChatGPT. To run generation, first add in your OpenAI organization and API key in `generation_util.py`. Then, use the following command to generate using ChatGPT:
```
sh ./generation_scripts/run_generate_cbg.sh
```
Alternatively, access our generated CBG letters stored in `./generated_letters/chatgpt/cbg/all_2_para_w_chatgpt.csv`