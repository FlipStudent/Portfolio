import sys

from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, GenerationConfig
import torch
from data import get_iam_dataset, ImageDataset
from tqdm import tqdm
from train import compute_score
from evaluate import load

BATCH_SIZE = 20


def image_to_text(model, processor: TrOCRProcessor, generation_config, dataloader, labels_available, device):
    """
    Loads data and saves model output to output.txt
    :param model: the model
    :param processor: processes model output
    :param generation_config: generation config
    :param dataloader: loads the data
    :param labels_available: if the data has labels, add them to the output and print CER and WER
    :param device: device where validation is run on
    :return:
    """
    model.eval()
    cer_metric = load("cer")
    wer_metric = load("wer")
    cer = 0.0
    wer = 0.0
    for batch in tqdm(dataloader):
        generated_ids = model.generate(batch["pixel_values"].to(device), generation_config=generation_config)
        output_strings = processor.batch_decode(generated_ids, skip_special_tokens=True)
        with open("output.txt", 'w') as file:
            for idx in range(len(output_strings)):
                file.write(batch["filenames"][idx] + "\n" +
                           output_strings[idx] + "\n")
                if labels_available:
                    file.write(batch["text_labels"][idx] + "\n")
                file.write("\n")

        if labels_available:
            cer += compute_score(cer_metric, generated_ids, batch["labels"], processor)
            wer += compute_score(wer_metric, generated_ids, batch["labels"], processor)

    if labels_available:
        cer /= len(dataloader)
        wer /= len(dataloader)
        print(f"CER: {cer*100:.2f}% | WER: {wer*100:.2f}%")


if __name__ == "__main__":
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    if len(sys.argv) < 2:
        _, val_dataloader = get_iam_dataset(processor, BATCH_SIZE)
        labels_available = True
    else:
        directory = sys.argv[1]
        print(f"Loading images from: {directory}...")
        dataset = ImageDataset(directory, None, processor)
        val_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        labels_available = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = VisionEncoderDecoderModel.from_pretrained("models/final_IAM/model_best", local_files_only=True).to(device)
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
    image_to_text(model, processor, generation_config, val_dataloader, labels_available, device)