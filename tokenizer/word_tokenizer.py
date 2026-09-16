class WordTokenizer:

    def __init__(self, text):

        self.unk_token = "<UNK>"

        # Split text into words
        words = text.split()

        # Create vocabulary
        self.words = sorted(set(words))

        if self.unk_token not in self.words:
            self.words.append(self.unk_token)

        self.word_to_id = {
            word: i
            for i, word in enumerate(self.words)
        }

        self.id_to_word = {
            i: word
            for i, word in enumerate(self.words)
        }

        self.unk_id = self.word_to_id[self.unk_token]

        self.vocab_size = len(self.words)


    def encode(self, text):

        words = text.split()

        return [
            self.word_to_id.get(
                word,
                self.unk_id
            )
            for word in words
        ]


    def decode(self, tokens):

        return " ".join(
            self.id_to_word[token]
            for token in tokens
        )