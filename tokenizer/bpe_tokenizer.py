import heapq
from collections import defaultdict


class BPETokenizer:
    """
    Byte-pair-encoding tokenizer over raw characters (with a handful of
    atomic special tokens for conversation structure).

    Training uses an incremental, heap-driven merge algorithm: instead
    of rescanning every adjacent pair in the whole symbol stream on
    every single merge (O(num_merges * text_length), which is what
    made fitting this tokenizer the slow part of both pretrain.py and
    train.py), it tracks pair counts and their occurrence positions in
    a doubly linked list and only touches the neighborhood of a pair
    each time it's merged. That makes each merge step roughly
    proportional to how often the merged pair occurs rather than to
    the length of the whole corpus, so total training time scales close
    to O(text_length log text_length) instead of O(text_length *
    num_merges). Encode/decode and the on-disk (tokens, merge_rules)
    format are unchanged, so existing checkpoints still load fine.
    """

    @classmethod
    def from_saved(cls, tokens, merge_rules):
        """
        Rebuild a tokenizer from a checkpoint's saved vocabulary and
        merge rules, without retraining. This is what generation code
        should use, so there's exactly one implementation of
        encode/decode instead of a second hand-copied one drifting out
        of sync with this file.
        """
        tokenizer = cls.__new__(cls)

        tokenizer.special_tokens = [
            "<UNK>",
            "<USER>",
            "<ASSISTANT>",
            "<END>"
        ]
        tokenizer.unk_token = "<UNK>"

        tokenizer.tokens = list(tokens)

        tokenizer.token_to_id = {
            token: index
            for index, token in enumerate(tokenizer.tokens)
        }

        tokenizer.id_to_token = {
            index: token
            for token, index in tokenizer.token_to_id.items()
        }

        tokenizer.unk_id = tokenizer.token_to_id[tokenizer.unk_token]

        tokenizer.merge_rules = [
            tuple(pair) for pair in merge_rules
        ]

        # Rank of each merge = its index in the learned order (lower
        # applies first). Built once here so encode() can look up
        # "does this pair merge, and how early" in O(1) instead of
        # walking the whole merge_rules list.
        tokenizer.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(tokenizer.merge_rules)
        }

        tokenizer.vocab_size = len(tokenizer.tokens)

        return tokenizer

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
        # Learn BPE merge rules (incremental heap algorithm).
        # --------------------------------------------------

        self.merge_rules = self._learn_merges(symbols, num_merges)

        # See from_saved() for what this is for.
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merge_rules)
        }

        self.vocab_size = len(
            self.tokens
        )

    # --------------------------------------------------
    # Incremental merge learning.
    # --------------------------------------------------

    def _learn_merges(self, symbols, num_merges):
        """
        Learns up to `num_merges` merge rules from `symbols`, updating
        self.tokens/token_to_id/id_to_token with any newly created
        merged tokens along the way. Returns the ordered list of
        merged (left, right) pairs, exactly as the old O(n * merges)
        implementation did -- this only changes *how* the merges are
        found, not what they are (modulo tie-breaking between
        equally-frequent pairs, which is not guaranteed to match the
        old dict-iteration-order tie-break, and doesn't need to: any
        max-count pair is an equally valid choice).
        """

        n = len(symbols)

        if n < 2:
            return []

        special_set = set(self.special_tokens)

        # Doubly linked list over live symbol positions, so a merge
        # only has to patch a handful of pointers instead of rebuilding
        # a list. `tok[i]` holds the current symbol living at node i
        # (updated in place when node i absorbs its right neighbor).
        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(0, n - 1))
        alive = [True] * n
        tok = list(symbols)

        pair_counts = defaultdict(int)
        pair_positions = defaultdict(set)

        def is_special(sym):
            return sym in special_set

        i = 0
        while i != -1:
            j = nxt[i]
            if j != -1 and not is_special(tok[i]) and not is_special(tok[j]):
                pair = (tok[i], tok[j])
                pair_counts[pair] += 1
                pair_positions[pair].add(i)
            i = j

        # Max-heap via negated counts. Entries go stale as merges
        # happen; staleness is caught by comparing against the live
        # pair_counts value when popped (lazy deletion), rather than
        # trying to remove/update entries buried in the heap.
        heap = [
            (-count, pair)
            for pair, count in pair_counts.items()
        ]
        heapq.heapify(heap)

        merge_rules = []
        merges_done = 0

        def bump(pair, delta, position, add):
            """Update pair_counts/pair_positions for `pair` and, if
            its count increased, push a fresh heap entry for it."""
            pair_counts[pair] += delta
            if add:
                pair_positions[pair].add(position)
            else:
                pair_positions[pair].discard(position)
            if delta > 0:
                heapq.heappush(heap, (-pair_counts[pair], pair))

        while merges_done < num_merges and heap:

            neg_count, pair = heapq.heappop(heap)
            count = -neg_count

            # Stale heap entry (count has since changed) -- skip it.
            if pair_counts.get(pair, 0) != count:
                continue

            # Same stopping rule as the original implementation: a
            # pair that only occurs once isn't worth merging.
            if count < 2:
                break

            left_sym, right_sym = pair
            merged_token = left_sym + right_sym

            if merged_token not in self.token_to_id:
                new_id = len(self.tokens)
                self.tokens.append(merged_token)
                self.token_to_id[merged_token] = new_id
                self.id_to_token[new_id] = merged_token

            merge_rules.append(pair)

            occurrences = list(pair_positions.get(pair, ()))
            pair_positions[pair] = set()
            pair_counts[pair] = 0

            for i in occurrences:

                if not alive[i]:
                    continue

                j = nxt[i]

                if j == -1 or not alive[j]:
                    continue

                # Symbol at i or j may have changed since this
                # occurrence was recorded (e.g. overlapping matches
                # like "aaa" merging pair ("a","a") -- once the first
                # two a's merge, this position's right neighbor is no
                # longer a lone "a"). Re-check before touching it.
                if tok[i] != left_sym or tok[j] != right_sym:
                    continue

                left_neighbor = prv[i]
                right_neighbor = nxt[j]

                if (
                    left_neighbor != -1
                    and alive[left_neighbor]
                    and not is_special(tok[left_neighbor])
                ):
                    old_pair = (tok[left_neighbor], tok[i])
                    bump(old_pair, -1, left_neighbor, add=False)

                if (
                    right_neighbor != -1
                    and alive[right_neighbor]
                    and not is_special(tok[right_neighbor])
                ):
                    old_pair = (tok[j], tok[right_neighbor])
                    bump(old_pair, -1, j, add=False)

                # Merge j into i.
                tok[i] = merged_token
                alive[j] = False
                nxt[i] = right_neighbor
                if right_neighbor != -1:
                    prv[right_neighbor] = i

                if (
                    left_neighbor != -1
                    and alive[left_neighbor]
                    and not is_special(tok[left_neighbor])
                ):
                    new_pair = (tok[left_neighbor], tok[i])
                    bump(new_pair, 1, left_neighbor, add=True)

                if (
                    right_neighbor != -1
                    and alive[right_neighbor]
                    and not is_special(tok[right_neighbor])
                ):
                    new_pair = (tok[i], tok[right_neighbor])
                    bump(new_pair, 1, i, add=True)

            merges_done += 1

        return merge_rules

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

        # Apply the learned merges -- lowest merge-rank pair first,
        # same effective order as running each merge_rules entry
        # across the whole token list in turn, just without the O(n)
        # full-list copy that a naive per-rule pass costs. Mirrors the
        # linked-list + heap approach _learn_merges uses, except the
        # heap here is keyed by each pair's fixed learned rank instead
        # of a live frequency count, since encode() isn't discovering
        # merges, just replaying a known, fixed order of them.
        tokens = self._apply_merges(tokens)

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

    def _apply_merges(self, tokens):
        """
        Applies self.merge_rules to `tokens` in learned-rank order,
        using a doubly linked list over live positions and a heap
        keyed by (rank, position) so only the neighborhood of an
        actual merge is ever touched -- not a full pass per rule. This
        is what makes encoding a multi-million-character corpus (e.g.
        training/pretrain.py's "Encoding full corpus..." step)
        tractable instead of taking O(num_merges * len(tokens)).
        """

        n = len(tokens)

        if n < 2 or not self.merge_ranks:
            return tokens

        special_set = set(self.special_tokens)

        def is_special(sym):
            return sym in special_set

        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(0, n - 1))
        alive = [True] * n
        tok = list(tokens)

        heap = []

        def push_if_mergeable(i, j):
            if (
                j == -1
                or is_special(tok[i])
                or is_special(tok[j])
            ):
                return
            rank = self.merge_ranks.get((tok[i], tok[j]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        i = 0
        while i != -1:
            push_if_mergeable(i, nxt[i])
            i = nxt[i]

        while heap:

            rank, i = heapq.heappop(heap)

            if not alive[i]:
                continue

            j = nxt[i]

            if j == -1 or not alive[j]:
                continue

            # Stale entry: the symbols at i/j (or the rank they'd
            # merge at) have changed since this was pushed -- e.g. an
            # earlier, lower-rank merge already consumed one of them,
            # or overlapping matches like "aaa" shifted what's here.
            pair = (tok[i], tok[j])
            if self.merge_ranks.get(pair) != rank:
                continue

            merged_token = pair[0] + pair[1]

            left_neighbor = prv[i]
            right_neighbor = nxt[j]

            tok[i] = merged_token
            alive[j] = False
            nxt[i] = right_neighbor
            if right_neighbor != -1:
                prv[right_neighbor] = i

            if left_neighbor != -1:
                push_if_mergeable(left_neighbor, i)
            push_if_mergeable(i, right_neighbor)

        result = []
        i = 0
        while i != -1:
            result.append(tok[i])
            i = nxt[i]

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
