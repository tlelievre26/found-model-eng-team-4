"""Dummy training run to check W&B logging end to end.

Trains a tiny model on random data for a few seconds and logs train / eval
metrics through ``src.utils.wandb_logging``. Use it to confirm your W&B login
works and that runs show config, git commit, GPU type, time, peak memory, and
cost:

    python scripts/wandb_smoke_test.py
    WANDB_MODE=disabled python scripts/wandb_smoke_test.py   # no W&B, just prints
"""

import argparse

import torch
from torch import nn

from src.utils import wandb_logging as wl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab, dim = 1000, 256
    model = nn.Sequential(nn.Embedding(vocab, dim), nn.Linear(dim, vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    config = vars(args) | {"model": "toy-embedding-linear", "vocab": vocab, "dim": dim}
    with wl.start_run("wandb-smoke-test", config, tags=["smoke-test"]):
        tokens_seen = 0
        for step in range(1, args.steps + 1):
            x = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
            # Fake padding so "real tokens" differs from batch_size * seq_len.
            mask = torch.rand(x.shape, device=device) > 0.1
            logits = model(x)
            loss = nn.functional.cross_entropy(logits[mask], x[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()

            tokens_seen += int(mask.sum())
            if step % 10 == 0:
                wl.log_train(step, tokens_seen, loss=loss, lr=args.lr)
            if step % 50 == 0:
                with torch.no_grad():
                    acc = (logits.argmax(-1) == x)[mask].float().mean()
                wl.log_eval({"accuracy": acc})


if __name__ == "__main__":
    main()
