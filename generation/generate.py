import torch

from config import ModelConfig, GenerationConfig
from model.transformer import Transformer, LanguageModelHead, KVCache
from tokenizer.bpe_tokenizer import BPETokenizer


# =========================
# Device
# =========================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# =========================
# Load checkpoint
# =========================

checkpoint = torch.load(
    "model.pt",
    map_location=device
)

vocab_size = checkpoint["vocab_size"]

# Rebuild the exact tokenizer used at training time from the saved
# vocabulary + merge rules (single source of truth: tokenizer/bpe_tokenizer.py).
tokenizer = BPETokenizer.from_saved(
    checkpoint["tokenizer_tokens"],
    checkpoint["tokenizer_merge_rules"]
)

# Prefer the architecture the checkpoint says it was trained with, so an
# out-of-date local config.py can't silently build the wrong-shaped model.
model_config = checkpoint.get("model_config", {
    "context_length": ModelConfig.context_length,
    "embedding_dim": ModelConfig.embedding_dim,
    "num_heads": ModelConfig.num_heads,
    "num_layers": ModelConfig.num_layers,
    "dropout": ModelConfig.dropout,
})

context_length = model_config["context_length"]
num_layers = model_config["num_layers"]


# =========================
# Build model
# =========================

transformer = Transformer(
    vocab_size=vocab_size,
    context_length=context_length,
    embedding_dim=model_config["embedding_dim"],
    num_heads=model_config["num_heads"],
    num_layers=num_layers,
    dropout=model_config["dropout"],
    gradient_checkpointing=model_config.get("gradient_checkpointing", False)
).to(device)

transformer.load_state_dict(checkpoint["transformer"])

# Weight tying: the LM head reuses the transformer's token embedding
# weight by reference, so there is nothing separate to load here.
lm_head = LanguageModelHead(
    transformer.embedding.token_embedding.embedding.weight
).to(device)

transformer.eval()
lm_head.eval()


print("BPE model loaded.")
print("Device:", device)
print("Vocabulary size:", vocab_size)

end_id = tokenizer.token_to_id["<END>"]
user_id = tokenizer.token_to_id["<USER>"]
assistant_id = tokenizer.token_to_id["<ASSISTANT>"]


# =========================
# Sampling
# =========================

def get_banned_ngram_tokens(generated_ids, n):
    """
    Standard no-repeat-ngram blocking (the same idea HF's generate()
    uses): if the last n-1 generated tokens have already appeared
    together earlier in this response, followed by some token T, then
    T is banned as a candidate for the next token right now — it would
    recreate an n-gram that's already been produced.

    This is a stronger, more targeted fix for loops like
    "Crescent, Crescent, name Crescent is Crescent" than the
    repetition penalty alone: the penalty only discourages individual
    tokens, so it can still shuffle the same small set of words into a
    new-looking but still-looping order. Blocking whole repeated
    n-grams stops that directly.
    """
    if n <= 0 or len(generated_ids) < n - 1:
        return set()

    if n == 1:
        # Degenerate case: ban every token already used.
        return set(generated_ids)

    prefix = tuple(generated_ids[-(n - 1):])
    banned = set()

    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i + n - 1]) == prefix:
            banned.add(generated_ids[i + n - 1])

    return banned


def filter_logits(logits, top_k=None, top_p=None, min_p=None):
    """
    Restricts `logits` (1D, shape (vocab_size,)) to a candidate pool
    using up to three complementary truncation strategies, applied in
    order -- each only ever narrows what the previous one allowed:

    - top_k: keep only the k highest-scoring tokens.
    - top_p (nucleus): keep the smallest set of (remaining) tokens
      whose cumulative probability reaches top_p.
    - min_p: drop any (remaining) token whose probability is below
      min_p * (probability of the single most likely token) -- a
      simpler, self-scaling alternative to top_p that adapts to how
      peaked or flat the distribution is.

    Any of the three can be disabled by passing None/0.
    """
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, k)[0][-1]
        logits[logits < kth_value] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Always keep at least the single most likely token, even if
        # its own probability already exceeds top_p.
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove[sorted_indices] = sorted_indices_to_remove
        logits[indices_to_remove] = float("-inf")

    if min_p is not None and 0 < min_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        threshold = min_p * probs.max()
        logits[probs < threshold] = float("-inf")

    return logits


def sample_next_token(
    logits,
    generated_ids,
    *,
    temperature,
    top_k,
    top_p,
    min_p,
    repetition_penalty,
    no_repeat_ngram_size
):
    """
    logits: 1D tensor of shape (vocab_size,) for the next position.
    generated_ids: token ids produced so far in this response, used for
    n-gram blocking and the repetition penalty.
    """
    logits = logits.clone()

    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        banned = get_banned_ngram_tokens(generated_ids, no_repeat_ngram_size)

        if banned:
            banned_indices = torch.tensor(
                list(banned),
                dtype=torch.long,
                device=logits.device
            )
            candidate_logits = logits.clone()
            candidate_logits[banned_indices] = float("-inf")

            # Safety net: if blocking would leave no valid candidate at
            # all (every token banned), fall back to the unblocked
            # logits rather than sampling from an all -inf vector.
            if not torch.isinf(candidate_logits).all():
                logits = candidate_logits

    if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            if torch.isinf(logits[token_id]):
                continue
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    if temperature <= 0.0:
        return torch.argmax(logits).item()

    logits = logits / temperature
    logits = filter_logits(logits, top_k=top_k, top_p=top_p, min_p=min_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# =========================
# KV-cache generation
# =========================

def generate_response(prompt_tokens, max_new_tokens):
    """
    Generates up to `max_new_tokens` tokens continuing `prompt_tokens`,
    yielding one new token id at a time as soon as it's sampled.

    Uses a KVCache so, after an initial "prefill" pass over the whole
    prompt, each subsequent step only runs the transformer over the
    single newest token instead of recomputing attention over the
    entire growing sequence -- the classic and by far the biggest
    lever for making autoregressive decoding fast. The trade-off: the
    cache's cos/sin rotary table and causal mask are only sized up to
    context_length, so a cached conversation can't "slide" past that
    the way the old from-scratch-every-step version implicitly could;
    generation just stops if the total sequence would exceed it.
    """
    tokens = list(prompt_tokens)

    if len(tokens) > context_length:
        # Keep the most recent context_length tokens. Note this can
        # only happen if the prompt itself is longer than
        # context_length; chat history is trimmed separately, well
        # before this point, to avoid losing the current question.
        tokens = tokens[-context_length:]

    kv_cache = KVCache(num_layers)
    generated_ids = []

    with torch.inference_mode():

        x = torch.tensor([tokens], dtype=torch.long, device=device)
        transformer_output = transformer(x, kv_cache=kv_cache)
        logits = lm_head(transformer_output)
        next_token_logits = logits[0, -1, :]

        total_length = len(tokens)

        for _ in range(max_new_tokens):

            next_token = sample_next_token(
                next_token_logits,
                generated_ids,
                temperature=GenerationConfig.temperature,
                top_k=GenerationConfig.top_k,
                top_p=GenerationConfig.top_p,
                min_p=GenerationConfig.min_p,
                repetition_penalty=GenerationConfig.repetition_penalty,
                no_repeat_ngram_size=GenerationConfig.no_repeat_ngram_size
            )

            generated_ids.append(next_token)
            yield next_token

            if next_token == end_id:
                break

            total_length += 1
            if total_length >= context_length:
                break

            x = torch.tensor([[next_token]], dtype=torch.long, device=device)
            transformer_output = transformer(x, kv_cache=kv_cache)
            logits = lm_head(transformer_output)
            next_token_logits = logits[0, -1, :]


# =========================
# Multi-turn chat loop
# =========================
#
# Keeps a running <USER>/<ASSISTANT>/<END> history and feeds it back
# in on every turn, so follow-up questions have context. This model's
# context_length is small, so before every turn we drop the oldest
# whole turns (by actual token count, not just turn count) until the
# history plus the new prompt leaves enough room for a reply -- so a
# conversation can go on indefinitely, it just gradually "forgets"
# the earliest turns instead of ever overflowing the model's context
# window (which previously could truncate the *current* prompt off
# the front and feed the model a mangled input).
#
# Commands: /reset clears history, /quit or /exit ends the session.

# Reserve room for the model's reply so trimming doesn't hand the
# model a prompt that already fills the entire context window.
RESERVE_FOR_RESPONSE = max(16, context_length // 4)
HISTORY_BUDGET = max(context_length - RESERVE_FOR_RESPONSE, 1)


def build_prompt_tokens(history_turns, prompt):
    """
    Encodes history_turns + the new prompt, dropping the oldest
    turn(s) first, until the result fits within HISTORY_BUDGET tokens
    (or there's no history left to drop).
    """
    turns = list(history_turns)

    while True:
        history_text = "".join(turns)
        formatted_prompt = history_text + "<USER> " + prompt + " <ASSISTANT>"
        tokens = tokenizer.encode(formatted_prompt)

        if len(tokens) <= HISTORY_BUDGET or not turns:
            return tokens

        turns.pop(0)


print(
    f"\nChat ready. Type a message and press Enter "
    f"(/reset to clear history, /quit to exit).\n"
)

history_turns = []

while True:

    try:
        prompt = input("You: ").strip()
    except EOFError:
        break

    if not prompt:
        continue

    if prompt.lower() in ("/quit", "/exit"):
        break

    if prompt.lower() == "/reset":
        history_turns = []
        print("(conversation history cleared)\n")
        continue

    tokens = build_prompt_tokens(history_turns, prompt)

    print("Assistant: ", end="", flush=True)

    generated_ids = []

    for token_id in generate_response(tokens, GenerationConfig.max_new_tokens):

        generated_ids.append(token_id)

        if token_id == end_id:
            break

        # Special tokens shouldn't normally appear mid-response, but
        # guard against printing them raw if the model ever does emit
        # one (e.g. early in training before it's learned the format).
        if token_id not in (user_id, assistant_id):
            print(tokenizer.decode([token_id]), end="", flush=True)

    print("\n")

    generated_text = tokenizer.decode(generated_ids)

    if "<END>" in generated_text:
        generated_text = generated_text.split("<END>", 1)[0]

    generated_text = generated_text.replace("<ASSISTANT>", "")
    generated_text = generated_text.replace("<USER>", "")
    generated_text = generated_text.strip()

    turn_text = (
        "<USER> " + prompt + " <ASSISTANT> " + generated_text + " <END> "
    )
    history_turns.append(turn_text)
    history_turns = history_turns[-GenerationConfig.max_history_turns:]
