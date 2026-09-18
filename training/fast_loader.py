import math

import torch


class FastTensorLoader:
    """
    Drop-in replacement for torch.utils.data.DataLoader, used in place
    of it for TextDataset/PlainTextDataset (training/dataset.py,
    training/pretrain_dataset.py) specifically because both of those
    already hold their entire contents in memory as a plain Python list
    of same-shaped (x, y, loss_mask) tensors (`.samples`).

    A regular DataLoader re-enters Python for every single sample via
    `Dataset.__getitem__`, then rebuilds each batch through
    `default_collate` -- on every step of every epoch. For a model this
    small (a few hundred thousand parameters, (batch, 256, 256)-shaped
    activations), that per-sample Python/collate overhead -- not GPU
    compute -- is the dominant cost of an epoch: the matmuls themselves
    run in well under a millisecond, so hundreds of batches' worth of
    interpreter overhead alone can add up to tens of seconds.

    This stacks the whole dataset into three tensors once, up front,
    and produces each epoch's batches with a single slice of a shuffled
    index tensor -- no per-sample Python calls, no collate. If `device`
    is a CUDA device, the stacked tensors are also moved there once
    here, rather than once per batch inside the training loop, removing
    the repeated host -> device copy entirely.

    This only makes sense because a dataset built from a few-MB text
    file is itself only a few MB to, at most, a couple hundred MB --
    nowhere near a 12GB GPU's budget. Don't reuse this for a dataset
    that doesn't comfortably fit in memory (or GPU memory) all at once.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        device=None,
        pin_memory=False
    ):
        # Accept either a Dataset with a `.samples` list (TextDataset,
        # PlainTextDataset) or a plain list of (x, y, loss_mask) tuples
        # directly.
        samples = getattr(dataset, "samples", dataset)

        if len(samples) == 0:
            raise ValueError(
                "FastTensorLoader received an empty dataset -- nothing "
                "to batch."
            )

        self.x = torch.stack([sample[0] for sample in samples])
        self.y = torch.stack([sample[1] for sample in samples])
        self.loss_mask = torch.stack([sample[2] for sample in samples])

        if device is not None and device.type == "cuda":
            self.x = self.x.to(device)
            self.y = self.y.to(device)
            self.loss_mask = self.loss_mask.to(device)
        elif pin_memory:
            self.x = self.x.pin_memory()
            self.y = self.y.pin_memory()
            self.loss_mask = self.loss_mask.pin_memory()

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return max(1, math.ceil(self.x.shape[0] / self.batch_size))

    def __iter__(self):
        num_samples = self.x.shape[0]

        # Matches the original DataLoader(shuffle=True) behavior: draw
        # from torch's global RNG (seeded once via torch.manual_seed at
        # the top of train.py/pretrain.py) rather than a private one.
        if self.shuffle:
            index = torch.randperm(num_samples)
        else:
            index = torch.arange(num_samples)

        # Move the (small) index tensor to the data's device once per
        # epoch, not once per batch.
        index = index.to(self.x.device)

        for start in range(0, num_samples, self.batch_size):
            batch_index = index[start:start + self.batch_size]

            yield (
                self.x[batch_index],
                self.y[batch_index],
                self.loss_mask[batch_index],
            )
