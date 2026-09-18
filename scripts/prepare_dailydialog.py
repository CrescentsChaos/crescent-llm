"""
Converts DailyDialog's raw dialogues_text.txt into this project's
training.txt format: one conversation per line, turns tagged
<USER>/<ASSISTANT>/<END>, ready to drop straight into
data/raw/training.txt.

--- Getting the raw data ---

1. Download: http://yanran.li/files/ijcnlp_dailydialog.zip
2. Unzip it. Inside is a `train` folder containing `train.zip` (and
   `validation`/`test` folders the same way) -- unzip those too. Each
   gives you a `dialogues_text.txt` file: one line per conversation,
   utterances separated by the literal string " __eou__ ".

--- Usage ---

    python scripts/prepare_dailydialog.py \
        path/to/train/dialogues_text.txt \
        path/to/validation/dialogues_text.txt \
        path/to/test/dialogues_text.txt \
        data/raw/training.txt

You can pass just the train split's file if you want less data /
a faster first run, or all three splits to use everything (~13,118
conversations total) -- this project's own train.py does its own
random train/validation split of whatever ends up in training.txt, so
there's no need to keep DailyDialog's original splits separate.

This OVERWRITES the output file, so back up your current
data/raw/training.txt first if you want to keep any of your existing
hand-written examples -- or just append a copy of the old file's
lines onto the converted output afterward, since the format is the
same.
"""

import re
import sys


# DailyDialog utterances often have stray spaces before punctuation
# (an artifact of how the dataset was tokenized when it was built),
# e.g. "Say , Jim , how about going" -- clean that up so the model is
# learning normal spacing instead of that quirk.
def clean_utterance(text):
    text = text.strip()
    text = re.sub(r"\s+([,.!?;:'])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def convert_file(path):
    """
    Reads one dialogues_text.txt and returns a list of already-
    formatted "<USER> ... <ASSISTANT> ... <END> ..." conversation
    strings, one per input line (dialogue). Utterances are paired up
    turn-by-turn: 1st -> <USER>, 2nd -> <ASSISTANT>, 3rd -> <USER>,
    4th -> <ASSISTANT>, etc. A trailing unpaired utterance (odd count)
    is dropped, since there's no reply for it to pair with.
    """
    conversations = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            utterances = [
                clean_utterance(u)
                for u in raw_line.split("__eou__")
            ]
            utterances = [u for u in utterances if u]

            if len(utterances) < 2:
                continue

            turns = []
            for i in range(0, len(utterances) - 1, 2):
                user_utt = utterances[i]
                assistant_utt = utterances[i + 1]
                turns.append(
                    f"<USER> {user_utt} <ASSISTANT> {assistant_utt} <END>"
                )

            if turns:
                conversations.append(" ".join(turns))

    return conversations


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/prepare_dailydialog.py "
            "<dialogues_text.txt> [more dialogues_text.txt ...] "
            "<output training.txt>"
        )
        sys.exit(1)

    *input_paths, output_path = sys.argv[1:]

    all_conversations = []

    for path in input_paths:
        converted = convert_file(path)
        print(f"{path}: {len(converted)} conversations converted")
        all_conversations.extend(converted)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_conversations:
            f.write(line + "\n")

    print(f"\nWrote {len(all_conversations)} conversations to {output_path}")


if __name__ == "__main__":
    main()
