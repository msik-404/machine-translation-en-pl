from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader

from collections import deque

EN_PATH = "data/en_data.txt"
PL_PATH = "data/pl_data.txt"

BATCH_SIZE = 64
MAX_LENGTH = 64


class EnPrestoDataset(Dataset):
    def __init__(self, tokenizer):
        self.path = EN_PATH
        text = []
        self.hash_counts = deque()
        with open(self.path, "r", encoding="utf-8") as f:
            for dialog in f:
                sentences = dialog.split(" # ")
                self.hash_counts.append(len(sentences))
                for sentence in sentences:
                    if len(sentence) == 0:
                        sentence = ""
                    elif sentence[-1] == "\n":
                        sentence = sentence[:-1]
                    text.append(sentence)
        self.inputs = tokenizer(
            text,
            max_length=MAX_LENGTH,  # avg sentence length is 36.8
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).input_ids.to("cuda")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


if __name__ == "__main__":
    model_checkpoint = "sdadas/mt5-base-translator-en-pl"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = EnPrestoDataset(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_checkpoint,
        device_map="auto",
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    curr_dialog_parts_left = dataset.hash_counts.popleft()
    curr_dialog = []
    counter = 0
    with open(PL_PATH, "w", encoding="utf-8") as f:
        for data in dataloader:
            generated_ids = model.generate(data, max_length=MAX_LENGTH)
            outputs = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for output in outputs:
                if curr_dialog_parts_left != 1:
                    curr_dialog.append(output)
                    curr_dialog.append(" # ")
                else:
                    curr_dialog.append(output)
                curr_dialog_parts_left -= 1
                if curr_dialog_parts_left == 0:
                    curr_dialog.append("\n")
                    f.write("".join(curr_dialog))
                    curr_dialog = []
                    curr_dialog_parts_left = dataset.hash_counts.popleft()

            counter += 1
            if counter % 10 == 0:
                print(counter*BATCH_SIZE, "/", len(dataset))
