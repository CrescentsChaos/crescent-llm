from pathlib import Path

text = Path("data/raw/training.txt").read_text(
    encoding="utf-8"
)

phrases = [
    "Python is",
    "Python is a",
    "Python is a programming",
    "Python is a programming language",
]

print("Phrase frequency:\n")

for phrase in phrases:
    count = text.count(phrase)

    print(
        f"{phrase!r}: {count}"
    )

print("\nExamples containing 'Python is':\n")

for line in text.splitlines():

    if "Python is" in line:

        print(line)