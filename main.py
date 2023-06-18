from transformers import pipeline
from torch.utils.data import Dataset

EN_PATH = "data/en_data.txt"
PL_PATH = "data/pl_data.txt"


class EnPrestoDataset(Dataset):
    def __init__(self):
        self.path = EN_PATH
        self.data = []
        with open(self.path, "r") as f:
            for line in f:
                for sentence in line.split(" # "):
                    if len(sentence) > 0:
                        if sentence[-1] == "\n":
                            sentence = sentence[:-1]
                        self.data.append(sentence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    dataset = EnPrestoDataset()
    max_length = len(max(dataset, key=lambda x: len(x)))
    model_checkpoint = "sdadas/mt5-base-translator-en-pl"
    translator = pipeline("translation", model=model_checkpoint, device=0)
    for out in translator(dataset, max_length=max_length):
        print(out)