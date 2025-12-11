"""
Sets up full training run, including pretraining on 'LAM' and 'selfmade' followed by finetuning on IAM.
"""

from transformers import TrOCRProcessor
from data import get_lam_dataset, get_iam_dataset, get_selfmade_dataset
from train import main

BATCH_SIZE = 20

if __name__ == "__main__":
    lam_save_dir = "pretrain_LAM-1"
    selfmade_save_dir = "pretrain_selfmade-1"
    final_dir = "pretrained-finetuned-augmented"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    lam_train_dataloader, lam_val_dataloader = get_lam_dataset(processor, BATCH_SIZE)
    iam_train_dataloader, iam_val_dataloader = get_iam_dataset(processor, BATCH_SIZE)
    selfmade_train_dataloader, selfmade_val_dataloader = get_selfmade_dataset(processor, BATCH_SIZE)
    print("Training on LAM...")
    main(processor, lam_save_dir, lam_train_dataloader, [lam_val_dataloader, iam_val_dataloader], num_epochs=10)
    print("Training on selfmade...")
    main(processor, selfmade_save_dir, selfmade_train_dataloader, [selfmade_val_dataloader, iam_val_dataloader], num_epochs=20, model_folder=lam_save_dir)
    print("Training on IAM...")
    main(processor, final_dir, iam_train_dataloader, [iam_val_dataloader], num_epochs=20, model_folder=selfmade_save_dir)
