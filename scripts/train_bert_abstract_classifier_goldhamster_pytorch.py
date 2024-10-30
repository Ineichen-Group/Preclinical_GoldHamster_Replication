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
### --------- Import text --------- ###
def read_text(pmid, docs_dir):
    txt_file = os.path.join(docs_dir, pmid + ".txt")
    with open(txt_file, "r") as text_file:
        text = text_file.read()
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
    return text


#######################################
### -------- Import Splits -------- ###
def import_splits(docs_dir, train_dev_test_dir, filename):
    tsv_file = filename[0:-5] + ".tsv"
    with open(os.path.join(tsv_file), "w") as writer:
        line = "PMID"
        for label in all_labels:
            line += "\t" + label
        line += "\tTEXT\n"
        writer.write(line)
        skipped = []
        doc_index = 0
        with open(os.path.join(train_dev_test_dir, filename), "r") as reader:
            lines = reader.readlines()
            for line in lines:
                pmid, str_labels = line.strip().split("\t")
                labels = str_labels.split(",")
                text = read_text(pmid, docs_dir)
                # exclude documents w/o text
                if len(text) == 0:
                    skipped.append(doc_index)
                    continue
                # print(pmid,labels,text)
                line = pmid
                for label in all_labels:
                    if label in labels:
                        line += "\t1"
                    else:
                        line += "\t0"
                line += "\t" + text + "\n"
                writer.write(line)
                doc_index += 1
        writer.close()
        return tsv_file, skipped


############################################
### --------- Pre-process data --------- ###
def pre_process_data(docs_dir, train_dev_test_dir, filename):
    # Import splits
    tsv_file, skipped = import_splits(docs_dir, train_dev_test_dir, filename)
    # Import data from tsv
    data = pd.read_csv(tsv_file, sep="\t")
    # Select required columns
    filters = []
    for label in all_labels:
        filters.append(label)
    filters.append("TEXT")
    data = data[filters]
    print(data)
    # Set your model output as categorical and save in new label col
    for label in all_labels:
        if label in data:
            data[label + "_label"] = pd.Categorical(data[label])
    # Transform your output to numeric
    for label in all_labels:
        if label in data:
            data[label] = data[label + "_label"].cat.codes
    return data, tsv_file, skipped


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


def build_model(transformer_model, config, data):
    num_labels_dict = {}
    for label in all_labels:
        num_classes = len(data[label + "_label"].value_counts())
        num_labels_dict[label] = num_classes
    model = BERTMultiLabelMultiClass(transformer_model, config, num_labels_dict)
    print(model)
    return model


#######################################
def save_history(history, filename="training_history.json"):
    with open(filename, "w") as f:
        json.dump(history, f)


### ------- Prepare Dataset ------- ###
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels  # labels is a dict of label tensors
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
        # Get labels
        label_item = {label: torch.tensor(self.labels[label][idx], dtype=torch.long) for label in all_labels}
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),  # remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_item
        }
        return item

    def __len__(self):
        return len(self.texts)


def prepare_dataset(data, tokenizer, max_length):
    texts = data['TEXT'].tolist()
    labels = {}
    for label in all_labels:
        labels[label] = data[label].values
    dataset = MultiLabelDataset(texts, labels, tokenizer, max_length)
    return dataset


#######################################
### ------- Train the model ------- ###
def train_model(
    model_name,
    model,
    data_train,
    data_dev,
    max_length,
    tokenizer,
    learning_rate,
    batch_size,
    epochs,
):

    # Prepare datasets
    train_dataset = prepare_dataset(data_train, tokenizer, max_length)
    dev_dataset = prepare_dataset(data_dev, tokenizer, max_length)
    # Prepare DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Set up loss functions
    loss_fns = {}
    for label in all_labels:
        loss_fns[label] = nn.CrossEntropyLoss()
    # Training loop
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        total_loss = 0
        total_correct = {label: 0 for label in all_labels}
        total_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {label: batch['labels'][label].to(device) for label in all_labels}
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            losses = []
            for label in all_labels:
                loss = loss_fns[label](outputs[label], labels[label])
                losses.append(loss)
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            total_examples += input_ids.size(0)
            # Compute accuracy
            with torch.no_grad():
                for label in all_labels:
                    preds = torch.argmax(outputs[label], dim=1)
                    correct = (preds == labels[label]).sum().item()
                    total_correct[label] += correct
        avg_loss = total_loss / total_examples
        print(f'Average training loss: {avg_loss}')
        for label in all_labels:
            acc = total_correct[label] / total_examples
            print(f'Training accuracy for {label}: {acc}')
        # Optionally, evaluate on validation set
    # Save the model
    model_name_clean = model_name.split("/")[1]
    print("Saving model to " + model_name_clean + "_model.pt")
    torch.save(model.state_dict(), model_name_clean + "_model.pt")


####################################
### ----- Train the model ------ ###
def train_bert_goldhamster2(
    model_name, docs_dir, train_dev_test_dir, train_file, dev_file, epochs, batch_size
):
    # import data
    data_train, tsv_file_train, skipped_train = pre_process_data(
        docs_dir, train_dev_test_dir, train_file
    )
    data_dev, tsv_file_dev, skipped_dev = pre_process_data(
        docs_dir, train_dev_test_dir, dev_file
    )
    # set up model
    transformer_model, config, max_length, tokenizer = setup_bert(model_name)
    # build model
    model = build_model(transformer_model, config, data_train)
    print("Training model " + model_name)
    # train model
    train_model(
        model_name,
        model,
        data_train,
        data_dev,
        max_length,
        tokenizer,
        learning_rate,
        batch_size,
        epochs,
    )


###########################################
### ----- Predict with the model ------ ###
def predict_with_model(model_name, model_name_clean, docs_dir, train_dev_test_dir, test_file, out_file):
    from torch.utils.data import DataLoader

    # import data
    data_test, tsv_file_test, skipped_test = pre_process_data(
        docs_dir, train_dev_test_dir, test_file
    )
    # set up model
    transformer_model, config, max_length, tokenizer = setup_bert(model_name)
    # build the model
    model = build_model(transformer_model, config, data_test)
    # Load model state dict
    print("loading model: " + model_name_clean + "_model.pt")
    model.load_state_dict(torch.load(model_name_clean + "_model.pt"))
    model.eval()
    # prepare test data
    test_dataset = prepare_dataset(data_test, tokenizer, max_length)
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
    print("Saving predictions to " + out_file)
    print_predictions(
        predictions, train_dev_test_dir, test_file, out_file, skipped_test
    )


def print_predictions(
    predictions, train_dev_test_dir, test_file, out_file, skipped_test
):
    print("saving predictions to: " + train_dev_test_dir + "/" + out_file)
    # print(len(predictions[label]))
    with open(os.path.join(train_dev_test_dir, out_file), "w") as writer:
        with open(os.path.join(train_dev_test_dir, test_file), "r") as reader:
            lines = reader.readlines()
            doc_index = 0
            pred_index = 0
            for line in lines:
                pmid, str_labels = line.strip().split("\t")
                list_labels = []
                if doc_index not in skipped_test:
                    for label in predictions:
                        arr_pred = predictions[label][pred_index]
                        # Assuming binary classification for each label
                        if arr_pred[1] > arr_pred[0]:
                            list_labels.append(label)
                    pred_index += 1
                doc_index += 1
                # print(pmid,list_labels)
                writer.write(pmid + "\t" + ",".join(list_labels) + "\n")
        writer.close()


def train_cross_validation(
    model_name, docs_dir, train_dev_test_dir, name, epochs, batch_size
):
    print(train_dev_test_dir)
    range_values = range(0, 10)
    model_name_clean = model_name.split("/")[1]
    for split in range_values:
        print("*** ", split, " ***")
        train_bert_goldhamster2(
            model_name,
            docs_dir,
            train_dev_test_dir,
            "train" + str(split) + ".txt",
            "dev" + str(split) + ".txt",
            epochs,
            batch_size,
        )
        predict_with_model(
            model_name,
            model_name_clean,
            docs_dir,
            train_dev_test_dir,
            "test" + str(split) + ".txt",
            model_name_clean + "_preds_" + str(split) + "_" + name + ".txt",
        )


def train_one_experiment(
    model_name, docs_dir, train_dev_test_dir, split, name, epochs, batch_size
):
    model_name_clean = model_name.split("/")[1]
    train_bert_goldhamster2(
        model_name,
        docs_dir,
        train_dev_test_dir,
        "train" + str(split) + ".txt",
        "dev" + str(split) + ".txt",
        epochs,
        batch_size,
    )
    predict_with_model(
        model_name,
        model_name_clean,
        docs_dir,
        train_dev_test_dir,
        "test" + str(split) + ".txt",
        model_name_clean + "_preds_" + str(split) + "_" + name + ".txt",
    )
    
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
    num_labels_dict = {label: 2 for label in all_labels}  # Assuming binary classification
    model = BERTMultiLabelMultiClass(transformer_model, config, num_labels_dict)
    return model

# New function to perform inference on unlabeled data
def predict_on_unlabeled_data(model_name, model_path, new_data_texts, out_file):
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
            for label in all_labels:
                arr_pred = predictions[label][idx]
                if arr_pred[1] > arr_pred[0]:  # Threshold can be adjusted
                    predicted_labels.append(label)
            writer.write(f"Text {idx}:\t" + ",".join(predicted_labels) + "\n")
    print(f"Predictions saved to {out_file}")

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
        "--n_epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--docs_dir", default=None, type=str, help="Path to the pubmed abstract files."
    )
    parser.add_argument(
        "--experiment_type",
        default="single",
        type=str,
        help="Should the training happen over all CV splits.",
    )
    parser.add_argument(
        "--train_dev_test_dir",
        default=None,
        type=str,
        help="Path to the CV data splits.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="michiyasunaga/BioLinkBERT-base",
        type=str,
        help="HuggingFace Model.",
    )

    args = parser.parse_args()
    epochs = args.n_epochs
    batch_size = args.batch_size
    docs_dir = args.docs_dir
    train_dev_test_dir = args.train_dev_test_dir
    model_name = args.model_name_or_path
    experiment_type = args.experiment_type

    if experiment_type == "single":
        print("Running single experiment...")
        train_one_experiment(
            model_name,
            docs_dir,
            train_dev_test_dir,
            best_split,
            "goldhamster",
            epochs,
            batch_size,
        )
    elif experiment_type == "inference_only":
        print("Running inference only...")
        model_name_clean = model_name.split("/")[1]
        predict_with_model(
            model_name,
            model_name_clean,
            docs_dir,
            train_dev_test_dir,
            "test" + str(best_split) + ".txt",
            model_name_clean + "_preds_" + str(best_split) + ".txt",
        )
    else:
        print("Running all CV trainings...")
        train_cross_validation(
            model_name, docs_dir, train_dev_test_dir, "goldhamster", epochs, batch_size
        )


if __name__ == "__main__":
    main()
