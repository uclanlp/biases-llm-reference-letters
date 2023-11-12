# Data Generation, Data Preprocessing, and Training Scripts For the Language Agency Classifier
Refer to the following instructions to generate the training data and training the Language Agency Classifier for our usage.

## Data Generation and Preprocessing
You may refer to the following steps to generate and preprocess the languae agency classification dataset. Alternatively, access our generated and preprocssed dataset stored in `/agency_classifier/agency_dataset/`

1. Generate the raw language agency classification dataset by prompting ChatGPT to rephrase a piece of original biography into an agentic version and a communal version. 

To run generation, , first add in your OpenAI organization and API key in `/agency_classifier/agency_generation_util.py`. Use the following command to run generation:
```
cd agency_classifier
sh run_generate.sh
```

The generated raw dataset will be stored in `/agency_classifier/agency_bios/BIOS_sampled_preprocessed.csv`

2. Then, split the generated raw file into train, test, and validation sets:
```
# Make sure you are still in the agency_classifier directory
sh run_split.sh
```

The processed datasets will be stored in `/agency_classifier/agency_dataset/`

## Training the Language Agency Classifier
You may refer to the following command to train the language agency classifier using the generated dataset. Alternatively, access our trained classifier checkpoint in Google Drive at:
> https://drive.google.com/drive/folders/119pIbWMrNLwOCxj9XwTBXA-kY02nytOc?usp=drive_link

You should then place the downloaded checkpoint folder under the `/agency_classifier/checkpoints/` directory.

To train the language agency classifier, run:
```
# Make sure you are still in the agency_classifier directory
sh run_train.sh
```