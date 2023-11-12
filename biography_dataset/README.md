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

2. Then, we conduct name swapping and gender swapping to i. mask out original persona information in the biography, and ii. create a gender-balanced biography dataset. Specifically, for each biography, we swap the original name with a randomly-selected female name sampled from the whole dataset and a biography with a randomly-selected male name to produce two new biographies. 

Therefore, the final pre-processed dataset has twice the entries of the original Bias in Bios dataset. Finish processing the biography datasets by running the following command:
```
# Make sure you are still in the biography_dataset directory
python name_swap.py
```

The final pre-processed biography datasets are grouped by occupations and genders, and are splitted into multiple csv files that are stored in the /biography_dataset/preprocessed_bios folder.