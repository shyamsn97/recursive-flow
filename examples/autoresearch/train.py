"""TinyStories GPT training script for autoresearch.

This file is intentionally small and hackable. Agents can change the model,
optimizer, schedule, batching, or sampling logic without understanding a large
training stack.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Simple knobs agents can tune
# ---------------------------------------------------------------------------

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch"
DATA_PATH = CACHE_DIR / "tinystories.txt"
DATA_BYTES = 20_000_000

TIME_BUDGET = 300
SEQ_LEN = 256
BATCH_SIZE = 64
EVAL_BATCHES = 32
VOCAB_SIZE = 256

N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.0

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 20
GRAD_CLIP = 1.0
SEED = 1337


def load_data() -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading TinyStories sample to {DATA_PATH}")
        remaining = DATA_BYTES
        with requests.get(DATA_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with DATA_PATH.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written = chunk[:remaining]
                    f.write(written)
                    remaining -= len(written)
                    if remaining <= 0:
                        break
    text = DATA_PATH.read_bytes().decode("utf-8", errors="ignore")
    if len(text) < 1_000_000:
        raise RuntimeError(f"TinyStories download looks too small: {len(text)} chars")
    return text


def encode(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    dropout: float = DROPOUT


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, BATCH_SIZE, SEQ_LEN, device)
        losses.append(model(x, y).item())
    model.train()
    return sum(losses) / len(losses)


def lr_for_step(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / max(1, WARMUP_STEPS)
    return LEARNING_RATE


def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)
        torch.set_float32_matmul_precision("high")

    text = load_data()
    tokens = encode(text)
    split = int(0.9 * len(tokens))
    train_data = tokens[:split]
    val_data = tokens[split:]

    config = GPTConfig()
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"config: {asdict(config)}")
    print(f"params: {num_params:,}")
    print(f"time_budget: {TIME_BUDGET}s")

    step = 0
    total_tokens = 0
    t_train = time.time()
    while time.time() - t_train < TIME_BUDGET:
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, device)
        for group in optimizer.param_groups:
            group["lr"] = lr_for_step(step)
        loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        step += 1
        total_tokens += BATCH_SIZE * SEQ_LEN
        if step == 1 or step % 50 == 0:
            elapsed = time.time() - t_train
            print(
                f"step {step:05d} | loss {loss.item():.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | elapsed {elapsed:.1f}s",
                flush=True,
            )

    val_loss = estimate_loss(model, val_data, device)
    val_bpb = val_loss / math.log(2)
    t_end = time.time()
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    )
    result = {
        "val_bpb": float(val_bpb),
        "val_loss": float(val_loss),
        "training_seconds": float(time.time() - t_train),
        "total_seconds": float(t_end - t_start),
        "peak_vram_mb": float(peak_vram_mb),
        "total_tokens_M": float(total_tokens / 1e6),
        "num_steps": int(step),
        "num_params_M": float(num_params / 1e6),
        "depth": int(config.n_layer),
    }
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "result.json"), "w") as f:
            json.dump(result, f, sort_keys=True)

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"training_seconds: {result['training_seconds']:.1f}")
    print(f"total_seconds:    {result['total_seconds']:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_tokens_M:   {result['total_tokens_M']:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {result['num_params_M']:.2f}")
    print(f"depth:            {config.n_layer}")


if __name__ == "__main__":
    main()
