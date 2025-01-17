#!/bin/bash
#SBATCH --job-name=preclin_goldhamster
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=32G   # Request 32 GB of memory per GPU
#SBATCH --output=inference_output_%j.log  # Save stdout to file
#SBATCH --error=inference_error_%j.log    # Save stderr to file

# Parameters
INFERENCE_SCRIPT="inference_bert_abstract_classifier_goldhamster_pytorch.py"
TUNED_MODEL="BioLinkBERT-base_1436_model.pt"
MODEL="michiyasunaga/BioLinkBERT-base"

PUBMED_CHUNK_ID=0
PUBMED_DATA_PATH="./full_pubmed_raw"
OUTPUTS_DATA_PATH="./model_predictions/neuro_pubmed"

FILE_FOR_INFERENCE="${PUBMED_DATA_PATH}/pmid_contents_chunk_${PUBMED_CHUNK_ID}.txt"

# Capture the start time
start_time=$(date +%s)

# Print the current configuration
echo "Running BERT model inference..."
echo "Model script: $MODEL_SCRIPT"

# Execute the Python script with specified parameters, and save both stdout and stderr to the same log file
python $INFERENCE_SCRIPT --model_name_or_path $MODEL --trained_model_path $TUNED_MODEL --pubmed_file $FILE_FOR_INFERENCE --output_dir $OUTPUTS_DATA_PATH

# Capture the end time
end_time=$(date +%s)

# Calculate the duration in minutes
duration=$(( (end_time - start_time) / 60 ))

# Inform the user that the training process has finished and display the elapsed time
echo "Inference complete. Duration: $duration minutes."