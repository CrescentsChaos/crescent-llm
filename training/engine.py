import math
import time

import torch
import torch.nn.functional as F


def train_model(
    transformer,
    lm_head,
    train_dataloader,
    val_dataloader,
    vocab_size,
    device,
    *,
    learning_rate,
    weight_decay,
    grad_clip_norm,
    num_epochs,
    warmup_steps,
    early_stopping_patience,
    checkpoint_path,
    checkpoint_extra,
    label="model",
    gradient_accumulation_steps=1,
    label_smoothing=0.0
):
    """
    Shared training loop used by both pretrain.py and train.py, so the
    optimizer setup, LR schedule, AMP handling, and early stopping are
    defined exactly once instead of drifting between two copies.

    `checkpoint_extra` is a dict of everything besides the transformer
    weights that should be saved alongside the best checkpoint (vocab
    size, tokenizer state, architecture config, etc).

    `gradient_accumulation_steps` lets the effective batch size
    (dataloader batch_size * this) exceed what actually fits in memory
    at once: gradients from several consecutive micro-batches are
    summed before a single optimizer step, rather than stepping after
    every micro-batch. 1 reproduces the original one-step-per-batch
    behavior exactly.

    `label_smoothing` (0.0-1.0) softens the cross-entropy targets;
    0.0 reproduces the original hard-target behavior exactly.

    Returns the best validation loss achieved.
    """

    decay_params = [
        p for p in transformer.parameters()
        if p.requires_grad and p.dim() >= 2
    ]
    no_decay_params = [
        p for p in transformer.parameters()
        if p.requires_grad and p.dim() < 2
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate
    )

    accumulation_steps = max(1, gradient_accumulation_steps)

    batches_per_epoch = max(1, len(train_dataloader))
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(batches_per_epoch / accumulation_steps)
    )
    total_steps = num_epochs * optimizer_steps_per_epoch
    warmup = min(warmup_steps, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup

        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    use_amp = device.type == "cuda"

    # Ampere (RTX 30-series and newer) runs bf16 at the same throughput
    # as fp16 on tensor cores, but bf16's wider exponent range means it
    # can't silently underflow the way fp16 can -- so it needs no loss
    # scaling. GradScaler(enabled=False) makes every scaler.* call below
    # a plain pass-through (scale/unscale become no-ops, step() just
    # calls optimizer.step()), so this reuses the exact same code path
    # for both cases rather than branching the training loop itself.
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and not use_bf16)

    def compute_loss(x, y, loss_mask):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp
        ):
            transformer_output = transformer(x)
            logits = lm_head(transformer_output)

            losses = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction="none",
                label_smoothing=label_smoothing
            )

            losses = losses.view(y.shape)

            mask_sum = loss_mask.sum().clamp(min=1)
            loss = (losses * loss_mask).sum() / mask_sum

        return loss

    print("\nStarting training...")
    training_start = time.perf_counter()
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        transformer.train()
        # Kept as a GPU tensor and only turned into a Python float once,
        # after the loop (see average_loss below) -- calling .item() on
        # every micro-batch forces the CPU to stop and wait for the GPU
        # to catch up right then, which serializes the two and stalls
        # exactly the kind of overlap that makes a model this small
        # (sub-millisecond compute per step) run close to GPU-bound
        # instead of Python/launch-overhead-bound.
        total_loss = torch.zeros((), device=device)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y, loss_mask) in enumerate(train_dataloader):

            loss = compute_loss(x, y, loss_mask)
            total_loss += loss.detach()

            # Divide by accumulation_steps so the summed gradient over
            # a full accumulation window matches what a single step on
            # the equivalent large batch would have produced.
            scaler.scale(loss / accumulation_steps).backward()

            is_last_micro_batch = (batch_idx + 1) == batches_per_epoch
            ready_to_step = (
                (batch_idx + 1) % accumulation_steps == 0
                or is_last_micro_batch
            )

            if ready_to_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    transformer.parameters(),
                    max_norm=grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        average_loss = (total_loss / batches_per_epoch).item()

        transformer.eval()
        validation_loss = torch.zeros((), device=device)

        with torch.no_grad():
            for x, y, loss_mask in val_dataloader:
                validation_loss += compute_loss(x, y, loss_mask).detach()

        average_validation_loss = (validation_loss / len(val_dataloader)).item()

        improved = average_validation_loss < best_validation_loss

        if improved:
            best_validation_loss = average_validation_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "transformer": transformer.state_dict(),
                    **checkpoint_extra
                },
                checkpoint_path
            )

            print(f"Best {label} saved!")
        else:
            epochs_without_improvement += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        current_lr = scheduler.get_last_lr()[0]

        # exp() of the average per-token cross-entropy loss: a more
        # intuitive "how surprised is the model, on average" number
        # than raw nll loss (a val loss of 1.5 vs 1.0 doesn't say much
        # on its own; a perplexity of ~4.5 vs ~2.7 is easier to feel).
        val_perplexity = math.exp(min(average_validation_loss, 20))

        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} "
            f"Train Loss: {average_loss:.4f} "
            f"Val Loss: {average_validation_loss:.4f} "
            f"Val PPL: {val_perplexity:.2f} "
            f"LR: {current_lr:.2e} "
            f"Time: {epoch_time:.3f}s"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"\nNo improvement for {early_stopping_patience} "
                f"epochs — stopping early at epoch {epoch + 1}."
            )
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - training_start

    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Best validation loss: {best_validation_loss:.4f}")

    return best_validation_loss
