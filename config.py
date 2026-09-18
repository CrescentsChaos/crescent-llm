class ModelConfig:
    """Architecture hyperparameters. Shared by pretraining and
    fine-tuning — if you change these after pretraining, the
    fine-tuning stage won't be able to load the pretrained weights
    (shapes won't match), so pick these once."""
    context_length = 128
    embedding_dim = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.1

    # Recompute each block's activations on the backward pass instead
    # of keeping them in memory (torch.utils.checkpoint). Slower per
    # step (~an extra forward pass) but cuts activation memory
    # substantially, which matters more as num_layers/embedding_dim/
    # context_length grow. Off by default since this toy-sized model
    # doesn't need it; flip on if you scale the model up and start
    # hitting out-of-memory errors.
    gradient_checkpointing = False


class PretrainConfig:
    """
    Stage 1 (optional): train on plain, unstructured text purely to
    learn general language patterns — grammar, common phrasing — before
    the model ever sees your Q&A data. Point corpus_path at a few MB of
    plain prose; simple modern text (e.g. the TinyStories dataset)
    works far better for this than a Q&A-style file like training.txt.

    This stage is genuinely optional. If you don't run it,
    training/train.py behaves exactly as before (trains from scratch
    on your Q&A data alone).
    """
    corpus_path = "data/raw/pretrain_corpus.txt"
    checkpoint_path = "pretrained.pt"

    # BPE merge-learning now uses an incremental heap-based algorithm
    # (see tokenizer/bpe_tokenizer.py) instead of rescanning the whole
    # symbol stream on every merge, so fitting the tokenizer is much
    # cheaper than it used to be. The vocabulary is still fit on a
    # bounded sample rather than the whole corpus -- that's about
    # keeping the *vocabulary* representative of common patterns, not
    # about runtime -- but for the same reason, keep the corpus itself
    # to a few MB, since the *encoding* step below does run over the
    # full corpus.
    tokenizer_fit_chars = 400_000
    bpe_merges = 500

    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 0.01
    grad_clip_norm = 1.0

    # Number of micro-batches accumulated before each optimizer step.
    # Effective batch size = batch_size * gradient_accumulation_steps,
    # without needing that many samples to fit in memory/on the GPU at
    # once. 1 disables accumulation (an optimizer step every batch,
    # same as before).
    gradient_accumulation_steps = 1

    # Smooths the cross-entropy target distribution instead of
    # training toward a one-hot target, which tends to keep the model
    # a bit less overconfident. 0.0 disables it (original behavior).
    label_smoothing = 0.0

    num_epochs = 20
    warmup_steps = 200
    early_stopping_patience = 5

    validation_ratio = 0.05
    seed = 42


class TrainConfig:
    """Stage 2: fine-tune on your Q&A data (training.txt)."""

    # Set this to PretrainConfig.checkpoint_path (or any pretrained
    # checkpoint path) to initialize from a pretrained model instead of
    # training from scratch. Leave as None to skip pretraining
    # entirely — the original from-scratch behavior.
    pretrained_checkpoint = None

    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 0.01
    grad_clip_norm = 1.0

    # See PretrainConfig.gradient_accumulation_steps.
    gradient_accumulation_steps = 1

    # See PretrainConfig.label_smoothing.
    label_smoothing = 0.0

    num_epochs = 150
    warmup_steps = 100          # linear LR warmup, then cosine decay
    early_stopping_patience = 15  # stop if val loss doesn't improve for N epochs

    validation_ratio = 0.10
    bpe_merges = 500  # only used when pretrained_checkpoint is None

    seed = 42


class GenerationConfig:
    """Sampling settings for generation/generate.py."""

    max_new_tokens = 150
    temperature = 0.8   # 0.0 = greedy (argmax); higher = more random
    top_k = 5            # sample only from the top-k most likely tokens; None/0 disables

    # Nucleus sampling: restrict candidates to the smallest set of
    # tokens whose cumulative probability reaches top_p. Applied after
    # top_k (so it can only narrow the candidate pool further). 1.0 or
    # None disables it.
    top_p = None

    # Discard any token whose probability is below
    # min_p * (probability of the single most likely token). A
    # simpler, self-scaling alternative/complement to top_k / top_p.
    # None disables it.
    min_p = None

    repetition_penalty = 1.3  # >1.0 discourages repeating already-used tokens
    no_repeat_ngram_size = 3  # block any token that would recreate an n-gram
                              # already seen in this response (0 disables)

    # Multi-turn chat mode (see generation/generate.py --chat): how
    # many most-recent <USER>/<ASSISTANT> turns to keep feeding back in
    # as context for the next reply. Older turns are dropped first if
    # the conversation would otherwise exceed the model's
    # context_length.
    max_history_turns = 6
