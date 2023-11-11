# Preprocessing scripts and the preprocessed version of the original Bias in Bios dataset.
Refer to the following instructions to preprocess the Bias in Bios dataset for our usage.
1. Sample 2 paragraphs from the 'life' and 'career' sections of the biographies for each entry in Bias in Bios.
```
cd biography_dataset
# Sampling for male biographies
sh run_sampling_m.sh
# Sampling for female biographies
sh run_sampling_f.sh
```