#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH --output=%A_%a.out  # %A is the job ID, %a is the array index
#SBATCH --error=%A_%a.err
#SBATCH --gres=gpu:1        # Reserve 1 GPU per job
#SBATCH --array=0-2         # Three jobs for three models
#SBATCH --time=45:00:00     # Set the maximum time for each job

# Define the array of models
model_names=("dmis-lab/biobert-v1.1" "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" "michiyasunaga/BioLinkBERT-base")

# Select the model based on the job array index
model_name=${model_names[$SLURM_ARRAY_TASK_ID]}

# Additional parameters for the training script
EPOCHS=10
DOCS_DIR="./pubmed_docs"
DATA_DIR="./corpus_annotations"
MODEL_SCRIPT="train_bert_abstract_classifier_goldhamster.py"

# Capture the start time
start_time=$(date +%s)

# Print the current configuration
echo "Running BERT model training..."
echo "Model: $model_name"
echo "Number of epochs: $EPOCHS"
echo "Documents directory: $DOCS_DIR"
echo "Training/Dev/Test directory: $DATA_DIR"
echo "Model script: $MODEL_SCRIPT"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Run the training for the selected model
python $MODEL_SCRIPT --experiment_type "parallel" --n_epochs $EPOCHS --docs_dir $DOCS_DIR --train_dev_test_dir $DATA_DIR --model_name_or_path $model_name

# Capture the end time
end_time=$(date +%s)

# Calculate the duration in minutes
duration=$(( (end_time - start_time) / 60 ))

# Inform the user that the training process has finished and display the elapsed time
echo "Training complete. Duration: $duration minutes."