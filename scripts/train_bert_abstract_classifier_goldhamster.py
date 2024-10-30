from transformers import TFBertModel, BertTokenizerFast, BertModel
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
import argparse
from tensorflow.keras.optimizers import legacy as legacy_optimizers
import logging
import sys
import json
import os
import pandas as pd

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
### -------- Setup HugginFace Model -------- ###
def setup_bert(model_name="dmis-lab/biobert-v1.1"):
    # Max length of tokens
    # 	max_length = 128
    max_length = 256
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    # Load the Transformers BERT model
    transformer_model = TFBertModel.from_pretrained(model_name, from_pt=True)

    config = transformer_model.config

    return transformer_model, config, max_length, tokenizer


#######################################
### ------- Build the model ------- ###


def build_model(transformer_model, config, max_length, data):
    # TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model# Load the MainLayer
    bert = transformer_model.layers[0]
    # Build your model input
    input_ids = Input(shape=(max_length,), name="input_ids", dtype="int32")
    inputs = {"input_ids": input_ids}
    # Load the Transformers BERT model as a layer in a Keras model
    bert_model = bert(inputs)[1]
    dropout = Dropout(config.hidden_dropout_prob, name="pooled_output")
    pooled_output = dropout(bert_model, training=False)
    # Then build your model output
    # one per output
    outputs = {}
    for label in all_labels:
        label_output = Dense(
            units=len(data[label + "_label"].value_counts()),
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            name=label,
        )(pooled_output)
        outputs[label] = label_output
    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=outputs, name="BERT_MultiLabel_MultiClass")
    # Take a look at the model
    model.summary()
    return model


#######################################
def save_history(history, filename="training_history.json"):
    with open(filename, "w") as f:
        json.dump(history.history, f)


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
    # Set an optimizer
    optimizer = legacy_optimizers.Adam(
        learning_rate, epsilon=1e-08, decay=0.01, clipnorm=1.0
    )
    # Set loss and metrics
    loss = {}
    for label in all_labels:
        loss[label] = CategoricalCrossentropy(from_logits=True)
    metric = {}
    for label in all_labels:
        metric[label] = CategoricalAccuracy("accuracy")
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    # trainig data
    x_train, y_train = prepare_x_y_dev_test(data_train, max_length, tokenizer)
    # validation data
    x_val, y_val = prepare_x_y_dev_test(data_dev, max_length, tokenizer)
    # Fit the model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
    )
    model_name_clean = model_name.split("/")[1]
    print("Saving model to " + model_name_clean + "_model.h5")
    model.save(model_name_clean + "_model.h5")
    save_history(history, model_name_clean + "_training_history.json")


def prepare_x_y_dev_test(data, max_length, tokenizer):
    test_ys = {}
    for label in all_labels:
        test_y = to_categorical(data[label])
        test_ys[label] = test_y
    test_x = tokenizer(
        text=data["TEXT"].to_list(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=True,
    )
    x = {"input_ids": test_x["input_ids"]}
    y = {}
    for label in all_labels:
        y[label] = test_ys[label]
    return x, y


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
    model = build_model(transformer_model, config, max_length, data_train)
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
# Ready test data
def predict_with_model(model_name, docs_dir, train_dev_test_dir, test_file, out_file):
    # import data
    data_test, tsv_file_test, skipped_test = pre_process_data(
        docs_dir, train_dev_test_dir, test_file
    )
    # set up model
    transformer_model, config, max_length, tokenizer = setup_bert()
    # prepare test data
    x, y = prepare_x_y_dev_test(data_test, max_length, tokenizer)
    # Load model
    model_name_clean = model_name.replace("/", "_")
    model = load_model(model_name_clean + "_model.h5")
    # Run predictions
    predictions = model.predict(x)
    print(model.summary())
    
    print("Saving predictions to " + out_file)
    print_predictions(
        predictions, train_dev_test_dir, test_file, out_file, skipped_test
    )


def print_predictions(
    predictions, train_dev_test_dir, test_file, out_file, skipped_test
):
    # print(len(predictions[label]))
    with open(os.path.join(train_dev_test_dir, out_file), "w") as writer:
        with open(os.path.join(train_dev_test_dir, test_file), "r") as reader:
            lines = reader.readlines()
            doc_index = 0
            for line in lines:
                pmid, str_labels = line.strip().split("\t")
                list_labels = []
                if doc_index not in skipped_test:
                    for label in predictions:
                        arr_pred = predictions[label][doc_index]
                        # print(arr_pred)
                        if arr_pred[1] > arr_pred[0]:
                            list_labels.append(label)
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
            docs_dir,
            train_dev_test_dir,
            "test" + str(split) + ".txt",
            model_name_clean + "_preds_" + str(split) + "_" + name + ".txt",
        )


def train_one_experiment(
    model_name, docs_dir, train_dev_test_dir, split, name, epochs, batch_size
):
    model_name_clean = model_name.replace("/", "_")
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
        docs_dir,
        train_dev_test_dir,
        "test" + str(split) + ".txt",
        model_name_clean + "_preds_" + str(split) + "_" + name + ".txt",
    )


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
        train_one_experiment(
            model_name,
            docs_dir,
            train_dev_test_dir,
            best_split,
            "goldhamster",
            epochs,
            batch_size,
        )
    else:
        train_cross_validation(
            model_name, docs_dir, train_dev_test_dir, "goldhamster", epochs, batch_size
        )

if __name__ == "__main__":
    main()
