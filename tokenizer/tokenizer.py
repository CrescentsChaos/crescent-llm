class CharacterTokenizer:
    def __init__(self, text):

        self.unk_token = "<UNK>"

        self.chars = sorted(list(set(text)))

        if self.unk_token not in self.chars:
            self.chars.append(self.unk_token)

        self.char_to_id = {
            char: i
            for i, char in enumerate(self.chars)
        }

        self.id_to_char = {
            i: char
            for i, char in enumerate(self.chars)
        }

        self.unk_id = self.char_to_id[self.unk_token]

        self.vocab_size = len(self.chars)


    def encode(self, text):

        return [
            self.char_to_id.get(
                char,
                self.unk_id
            )
            for char in text
        ]


    def decode(self, tokens):

        return "".join(
            self.id_to_char[token]
            for token in tokens
        )