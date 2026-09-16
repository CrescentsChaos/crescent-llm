class BPETokenizer:

    def __init__(
        self,
        text,
        num_merges=100
    ):

        self.special_tokens = [
            "<UNK>",
            "<USER>",
            "<ASSISTANT>",
            "<END>"
        ]

        self.unk_token = "<UNK>"

        # --------------------------------------------------
        # Build initial vocabulary.
        # --------------------------------------------------

        characters = set()

        for character in text:
            characters.add(character)

        self.tokens = (
            self.special_tokens
            + sorted(characters)
        )

        self.token_to_id = {
            token: index
            for index, token in enumerate(self.tokens)
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

        self.unk_id = self.token_to_id[
            self.unk_token
        ]

        # --------------------------------------------------
        # Convert text into BPE symbols.
        #
        # Special tokens remain atomic.
        # Everything else starts as characters.
        # Spaces are preserved.
        # --------------------------------------------------

        symbols = []

        i = 0

        while i < len(text):

            matched_special = None

            for special_token in self.special_tokens[1:]:

                if text.startswith(
                    special_token,
                    i
                ):

                    matched_special = special_token
                    break

            if matched_special:

                symbols.append(
                    matched_special
                )

                i += len(
                    matched_special
                )

            else:

                symbols.append(
                    text[i]
                )

                i += 1

        # --------------------------------------------------
        # Learn BPE merge rules.
        # --------------------------------------------------

        self.merge_rules = []

        for _ in range(num_merges):

            pair_counts = {}

            for i in range(
                len(symbols) - 1
            ):

                left = symbols[i]
                right = symbols[i + 1]

                # Never merge special tokens.
                if (
                    left in self.special_tokens
                    or right in self.special_tokens
                ):
                    continue

                pair = (
                    left,
                    right
                )

                pair_counts[pair] = (
                    pair_counts.get(
                        pair,
                        0
                    )
                    + 1
                )

            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=pair_counts.get
            )

            if pair_counts[best_pair] < 2:
                break

            self.merge_rules.append(
                best_pair
            )

            merged_token = (
                best_pair[0]
                + best_pair[1]
            )

            if (
                merged_token
                not in self.token_to_id
            ):

                new_id = len(
                    self.tokens
                )

                self.tokens.append(
                    merged_token
                )

                self.token_to_id[
                    merged_token
                ] = new_id

                self.id_to_token[
                    new_id
                ] = merged_token

            symbols = self._merge_pair(
                symbols,
                best_pair
            )

        self.vocab_size = len(
            self.tokens
        )

    # --------------------------------------------------
    # Merge a pair.
    # --------------------------------------------------

    def _merge_pair(
        self,
        tokens,
        pair
    ):

        result = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):

                result.append(
                    tokens[i]
                    + tokens[i + 1]
                )

                i += 2

            else:

                result.append(
                    tokens[i]
                )

                i += 1

        return result

    # --------------------------------------------------
    # Encode text.
    # --------------------------------------------------

    def encode(
        self,
        text
    ):

        # Start with characters and special tokens.
        tokens = []

        i = 0

        while i < len(text):

            matched_special = None

            for special_token in self.special_tokens[1:]:

                if text.startswith(
                    special_token,
                    i
                ):

                    matched_special = special_token
                    break

            if matched_special:

                tokens.append(
                    matched_special
                )

                i += len(
                    matched_special
                )

            else:

                tokens.append(
                    text[i]
                )

                i += 1

        # Apply the learned merges.
        for pair in self.merge_rules:

            tokens = self._merge_pair(
                tokens,
                pair
            )

        # Convert tokens to IDs.
        result = []

        for token in tokens:

            if token in self.token_to_id:

                result.append(
                    self.token_to_id[
                        token
                    ]
                )

            else:

                # Fallback to characters.
                for character in token:

                    if (
                        character
                        in self.token_to_id
                    ):

                        result.append(
                            self.token_to_id[
                                character
                            ]
                        )

                    else:

                        result.append(
                            self.unk_id
                        )

        return result

    # --------------------------------------------------
    # Decode token IDs.
    # --------------------------------------------------

    def decode(
        self,
        token_ids
    ):

        result = ""

        for token_id in token_ids:

            token = self.id_to_token.get(
                token_id,
                self.unk_token
            )

            result += token

        return result