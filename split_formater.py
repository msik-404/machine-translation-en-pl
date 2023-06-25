EN_PATH = "data/en_data.txt"
OUT_PATH = "data/out_en_data.txt"

if __name__ == "__main__":
    with open(EN_PATH, "r", encoding="utf-8") as f:
        with open(OUT_PATH, "w", encoding="utf-8") as o:
            for line in f:
                for sentence in line.split(" # "):
                    if len(sentence) > 0:
                        if sentence[-1] == "\n":
                            sentence = sentence[:-1]
                        o.write(sentence)
                        o.write("\n")
