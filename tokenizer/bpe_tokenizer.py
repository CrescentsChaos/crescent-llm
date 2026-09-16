from collections import Counter


class BPETokenizer:

    def __init__(self, text, num_merges=50):

        self.unk_token = "<UNK>"

        # Start with characters
        words = text.split()

        tokens = [
            list(word)
            for word in words
        ]

        # Initial vocabulary
        self.vocabulary = set()
        self.vocabulary.add(" ")

        for word_tokens in tokens:
            self.vocabulary.update(
                word_tokens
            )

        # Store learned merge rules
        self.merge_rules = []

        # Learn BPE merges
        for _ in range(num_merges):

            pairs = Counter()

            for word_tokens in tokens:

                pairs.update(
                    zip(
                        word_tokens,
                        word_tokens[1:]
                    )
                )

            if not pairs:
                break

            most_common_pair, _ = (
                pairs.most_common(1)[0]
            )

            self.merge_rules.append(
                most_common_pair
            )

            first, second = most_common_pair

            merged_token = first + second

            self.vocabulary.add(
                merged_token
            )

            # Apply merge inside every word
            for index in range(
                len(tokens)
            ):

                tokens[index] = (
                    self._merge_pair(
                        tokens[index],
                        most_common_pair
                    )
                )

        # Add unknown token
        self.vocabulary.add(
            self.unk_token
        )

        # Create token IDs
        self.tokens = sorted(
            self.vocabulary,
            key=lambda x: (len(x), x)
        )

        self.token_to_id = {
            token: token_id
            for token_id, token in enumerate(
                self.tokens
            )
        }

        self.id_to_token = {
            token_id: token
            for token, token_id
            in self.token_to_id.items()
        }

        self.unk_id = self.token_to_id[
            self.unk_token
        ]

        self.vocab_size = len(
            self.tokens
        )

    def _merge_pair(self, tokens, pair):

        first, second = pair

        merged_token = first + second

        new_tokens = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == first
                and tokens[i + 1] == second
            ):

                new_tokens.append(
                    merged_token
                )

                i += 2

            else:

                new_tokens.append(
                    tokens[i]
                )

                i += 1

        return new_tokens

    def encode(self, text):

        words = text.split()

        result = []

        for word in words:

            tokens = list(word)

            # Apply learned merges
            for pair in self.merge_rules:

                tokens = self._merge_pair(
                    tokens,
                    pair
                )

            for token in tokens:

                result.append(
                    self.token_to_id.get(
                        token,
                        self.unk_id
                    )
                )

            # Preserve the space
            result.append(
                self.token_to_id.get(
                    " ",
                    self.unk_id
                )
            )

        # Remove final space
        if result:
            result.pop()

        return result

    def decode(self, token_ids):

        tokens = [
            self.id_to_token.get(
                token_id,
                self.unk_token
            )
            for token_id in token_ids
        ]

        return "".join(tokens)