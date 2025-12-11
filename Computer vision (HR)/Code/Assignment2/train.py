# Created with the help of the following GitHub repo:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb
import torch
from data import get_iam_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from evaluate import load
from transformers import GenerationConfig
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# hyperparameters and preset variables
BATCH_SZE = 20
NUM_EPOCHS = 20
LR = 5e-5
WD = 1e-2
DROPOUT = 0.0
PLT_DIR = "/plots"
BEST_MDL_DIR = "/model_best"
FINAL_MDL_DIR = "/model_final"


def plot_results(data_lists, ylabel, fn, save_dir):
    for data, label in data_lists:
        np.save(save_dir + PLT_DIR + "/" + label.replace(" ", "") + ".npy", data)
        plt.plot(data, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_dir + PLT_DIR + "/" + fn)
    plt.clf()


def compute_score(metric, pred_ids, label_ids, processor):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    score = metric.compute(predictions=pred_str, references=label_str)
    return score


def train(model: VisionEncoderDecoderModel, generation_config, processor, train_dataloader, val_dataloaders, num_epochs, device, save_dir):
    """
    Train loop
    :param model: model
    :param generation_config: generation config used for generating strings from model output
    :param processor: generates strings from model output
    :param train_dataloader: loads training data
    :param val_dataloaders: list of objects that loads validation data, allows for validating on more than 1 dataset
    :param num_epochs: amount of epochs
    :param device: device to run training on
    :param save_dir: where the model is saved
    :return:
    """
    cer_metric = load("cer")
    wer_metric = load("wer")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    train_losses = []
    valid_losses = []
    cer_scores = []
    wer_scores = []
    for _ in range(len(val_dataloaders)):
        valid_losses.append([])
        cer_scores.append([])
        wer_scores.append([])
    best_valid_loss = torch.inf
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            batch = {
                "pixel_values": batch["pixel_values"],
                "labels": batch["labels"]
            }
            # get the inputs
            for k, v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # evaluate
        model.eval()
        valid_loss = 0.0
        valid_cer = 0.0
        valid_wer = 0.0
        own_data_valid_loss = 0.0
        own_data_cer = 0.0
        own_data_wer = 0.0
        for idx in range(len(val_dataloaders)):
            with ((torch.no_grad())):
                for batch in tqdm(val_dataloaders[idx]):
                    batch = {
                        "pixel_values": batch["pixel_values"],
                        "labels": batch["labels"]
                    }

                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    # Forward pass to calculate loss
                    outputs = model(**batch)
                    loss = outputs.loss
                    valid_loss += loss.item()

                    # run batch generation
                    outputs = model.generate(batch["pixel_values"].to(device), generation_config=generation_config)
                    # compute metrics
                    cer = compute_score(cer_metric, outputs, batch["labels"], processor)
                    valid_cer += cer
                    wer = compute_score(wer_metric, outputs, batch["labels"], processor)
                    valid_wer += wer

            valid_loss = valid_loss / len(val_dataloaders[idx])
            valid_losses[idx].append(valid_loss)
            valid_cer = valid_cer / len(val_dataloaders[idx])
            cer_scores[idx].append(valid_cer)
            valid_wer = valid_wer / len(val_dataloaders[idx])
            wer_scores[idx].append(valid_wer)
            if idx == 0:
                own_data_valid_loss = valid_loss
                own_data_cer = valid_cer
                own_data_wer = valid_wer
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    model_dir = save_dir + BEST_MDL_DIR
                    model.save_pretrained(model_dir)
                    # torch.save({
                    #     "epoch": epoch,
                    #     "optim": optimizer.state_dict()
                    # }, f"{model_dir}/checkpoint.pt")

        print(f"epoch: {epoch} | train loss: {train_loss:.2f} | valid loss: {own_data_valid_loss:.2f} | CER: {own_data_cer*100:.2f}% | WER: {own_data_wer*100:.2f}%", flush=True)
        loss_list = [(train_losses, "Train loss"), (valid_losses[0], "Validation loss")]
        if len(valid_losses) == 2:
            loss_list.append((valid_losses[1], "IAM Validation loss"))
        plot_results(loss_list, ylabel="Loss", fn="loss_plot", save_dir=save_dir)
        error_scores = [(cer_scores[0], "CER"), (wer_scores[0], "WER")]
        if len(cer_scores) == 2:
            error_scores.append((cer_scores[1], "IAM CER"))
            error_scores.append((wer_scores[1], "IAM WER"))
        plot_results(error_scores, ylabel="Error", fn="error_plot", save_dir=save_dir)

    # model_dir = save_dir + FINAL_MDL_DIR
    # model.save_pretrained(model_dir)
    # torch.save({
    #     "epoch": NUM_EPOCHS,
    #     "optim": optimizer.state_dict()
    # }, f"{model_dir}/checkpoint.pt")

    return model


def main(processor, save_dir, train_dataloader, val_dataloaders, num_epochs, model_folder=None):
    """
    Prepares model and dataloaders for training
    :param processor: processes model output
    :param save_dir: where model is saved
    :param train_dataloader: loads training data
    :param val_dataloaders: list of objects that loads validation data, allows for validating on more than 1 dataset
    :param num_epochs: amount of epochs
    :param model_folder: folder to load pretrained model from
    :return:
    """
    save_dir = "models/" + save_dir
    model_path = None
    if model_folder:
        model_path = "models/" + model_folder + BEST_MDL_DIR

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + PLT_DIR):
        os.makedirs(save_dir + PLT_DIR)
    if not os.path.exists(save_dir + BEST_MDL_DIR):
        os.makedirs(save_dir + BEST_MDL_DIR)
    if not os.path.exists(save_dir + FINAL_MDL_DIR):
        os.makedirs(save_dir + FINAL_MDL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    if model_folder:
        print(f"Loading model: {model_folder}")
        model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True).to(device)
    else:
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").to(device)

    model.encoder.embeddings.dropout.p = DROPOUT
    for layer in model.encoder.encoder.layer:
        layer.attention.attention.dropout.p = DROPOUT
        layer.attention.output.dropout.p = DROPOUT
        layer.output.dropout.p = DROPOUT

    generation_config = GenerationConfig(
        max_length=64,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        num_beams=4,
        decoder_start_token_id=processor.tokenizer.cls_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.sep_token_id
    )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    return train(model, generation_config, processor, train_dataloader, val_dataloaders, num_epochs, device, save_dir)


if __name__ == "__main__":
    save_dir = "IAM-nothing"
    label_fn = "data/IAM/iam_lines_gt.txt"
    img_dir = "data/IAM/img"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    train_dataloader, val_dataloader = get_iam_dataset(processor, BATCH_SZE)
    main(processor, save_dir, train_dataloader, [val_dataloader], NUM_EPOCHS)
