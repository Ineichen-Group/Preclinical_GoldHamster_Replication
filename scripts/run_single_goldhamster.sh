#!/bin/bash
#SBATCH --job-name=preclin_goldhamster
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --mem-per-gpu=32G   # Request 32 GB of memory per GPU
#SBATCH --output=training_output_%j.log  # Save stdout to file
#SBATCH --error=training_error_%j.log    # Save stderr to file

# Parameters
EPOCHS=10
DOCS_DIR="./pubmed_docs"
DATA_DIR="./corpus_annotations"
MODEL_SCRIPT="train_bert_abstract_classifier_goldhamster_pytorch.py"
MODEL="michiyasunaga/BioLinkBERT-base"

# Capture the start time
start_time=$(date +%s)

# Print the current configuration
echo "Running BERT model training..."
echo "Number of epochs: $EPOCHS"
echo "Documents directory: $DOCS_DIR"
echo "Training/Dev/Test directory: $DATA_DIR"
echo "Model script: $MODEL_SCRIPT"

# Execute the Python script with specified parameters, and save both stdout and stderr to the same log file
python $MODEL_SCRIPT --experiment_type "all" --n_epochs $EPOCHS --docs_dir $DOCS_DIR --train_dev_test_dir $DATA_DIR

# Capture the end time
end_time=$(date +%s)

# Calculate the duration in minutes
duration=$(( (end_time - start_time) / 60 ))

# Inform the user that the training process has finished and display the elapsed time
echo "Training complete. Duration: $duration minutes."