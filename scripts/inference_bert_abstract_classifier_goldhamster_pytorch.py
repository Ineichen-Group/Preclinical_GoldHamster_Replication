from transformers import BertTokenizerFast, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import logging
import sys
import json
import os
import pandas as pd
from torch.utils.data import DataLoader

all_labels = [
    "in_silico",
    "organs",
    "other",
    "human",
    "in_vivo",
    "invertebrate",
    "primary_cells",
    "immortal_cell_line",
]
learning_rate = 1e-04
logger = logging.getLogger(__name__)

#######################################
### -------- Setup HuggingFace Model -------- ###
def setup_bert(model_name="dmis-lab/biobert-v1.1"):
    # Max length of tokens
    max_length = 256
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    # Load the Transformers BERT model
    transformer_model = BertModel.from_pretrained(model_name)

    config = transformer_model.config

    return transformer_model, config, max_length, tokenizer

#######################################
### ------- Build the model ------- ###
class BERTMultiLabelMultiClass(nn.Module):
    def __init__(self, transformer_model, config, num_labels_dict):
        super(BERTMultiLabelMultiClass, self).__init__()
        self.bert = transformer_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleDict()
        for label in all_labels:
            num_classes = num_labels_dict[label]
            self.classifiers[label] = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # get pooled output
        pooled_output = self.dropout(pooled_output)
        logits = {}
        for label in all_labels:
            logits[label] = self.classifiers[label](pooled_output)
        return logits
    

###########################################
### ----- Inference on unlabelled data ------ ###
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),  # remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        return item

    def __len__(self):
        return len(self.texts)

# Adjusted build_model function for inference
def build_model_for_inference(transformer_model, config):
    num_labels_dict = {label: 2 for label in all_labels}  # Assuming binary classification TODO: should be dataset dependent?
    model = BERTMultiLabelMultiClass(transformer_model, config, num_labels_dict)
    return model

# New function to perform inference on unlabeled data
def predict_on_unlabeled_data(model_name, model_path, new_data_ids, new_data_texts, out_file):
    # Load model components
    transformer_model, config, max_length, tokenizer = setup_bert(model_name)
    # Build the model for inference
    model = build_model_for_inference(transformer_model, config)
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Prepare dataset and dataloader
    test_dataset = UnlabeledDataset(new_data_texts, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=32)
    # Run predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    predictions = {label: [] for label in all_labels}
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            for label in all_labels:
                preds = F.softmax(outputs[label], dim=1)  # get probabilities
                predictions[label].extend(preds.cpu().numpy())
    # Process and save predictions
    with open(out_file, "w") as writer:
        for idx, text in enumerate(new_data_texts):
            predicted_labels = []
            pmid = new_data_ids[idx]
            for label in all_labels:
                arr_pred = predictions[label][idx]
                if arr_pred[1] > arr_pred[0]:  # Threshold can be adjusted
                    predicted_labels.append(label)
            writer.write(f"Text {pmid}:\t" + ",".join(predicted_labels) + "\n")
    print(f"Predictions saved to {out_file}")
    
def read_pubmed_chunk(file_path, headers):
    print(f"Reading file {file_path}")
    all_data_df = pd.read_csv(file_path, sep=r'\^!\^', names=headers, engine='python')  # Change 'sep' if files use a different delimiter
    
    df_empty_abstracts = all_data_df[all_data_df['AbstractText'].isna()]
    df_non_empty_abstracts = all_data_df[all_data_df['AbstractText'].notna()]
    
    print(f"Loading {len(df_non_empty_abstracts)} pmids, ignoring {len(df_empty_abstracts)} due to empty abstracts.")
    return list(df_non_empty_abstracts['PMID']),  list(df_non_empty_abstracts['AbstractText'])

def main():
    best_split = 1

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Train and evaluate model.")
  
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="michiyasunaga/BioLinkBERT-base",
        help="The name or path of the HuggingFace model to use. For example, 'michiyasunaga/BioLinkBERT-base'."
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default="BioLinkBERT-base_1436_model.pt",
        help="Path to the fine-tuned .pt file of the HuggingFace model."
    )
    parser.add_argument(
        "--pubmed_file",
        type=str,
        default="./data/full_pubmed_raw/pmid_contents_chunk_0.txt",
        help="File with PubMed content."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_predictions/neuro_pubmed",
        help="Directory path where prediction outputs will be saved."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File name to save predictions, e.g. ./model_predictions/neuro_pubmed/predictions_chunk_0.txt."
    )

    args = parser.parse_args()
   
    model_name = args.model_name_or_path
    model_path = args.trained_model_path
    model_name_clean = model_name.split("/")[1]
    
    headers = ["PMID", "Year", "Journal", "Title", "AbstractText", "DOI"]
    input_file_path = args.pubmed_file
    new_data_ids, new_data_texts = read_pubmed_chunk(input_file_path, headers)
    
    outputs_data_path = args.output_dir
    out_file = args.output_file
    if not out_file:
        file_ending = input_file_path.split("_")[-1]
        out_file = f"{outputs_data_path}/{model_name_clean}_pred_pubmed_chunk_{file_ending}"
    predict_on_unlabeled_data(model_name, model_path, new_data_ids, new_data_texts, out_file)

if __name__ == "__main__":
    main()
