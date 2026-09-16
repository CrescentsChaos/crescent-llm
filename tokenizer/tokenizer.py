class CharacterTokenizer:

    def __init__(self, text):
        # Create vocabulary from all unique characters
        self.chars = sorted(list(set(text)))

        # Character -> integer
        self.char_to_id = {
            char: i
            for i, char in enumerate(self.chars)
        }

        # Integer -> character
        self.id_to_char = {
            i: char
            for i, char in enumerate(self.chars)
        }

        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Convert text into token IDs."""
        return [
            self.char_to_id[char]
            for char in text
        ]

    def decode(self, tokens):
        """Convert token IDs back into text."""
        return "".join(
            self.id_to_char[token]
            for token in tokens
        )