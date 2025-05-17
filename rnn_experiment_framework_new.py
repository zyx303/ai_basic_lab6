"""
Advanced RNN Experiment Framework
================================
Upgraded character‑level text generator with an embedding layer, multi‑layer
LSTM, dropout regularisation and tunable hyper‑parameters for improved model
capacity and performance.  Run
    python rnn_experiment_framework.py --help
for a full list of options.
"""



import argparse
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange  # Add tqdm for progress visualization

# -----------------------------------------------------------------------------
# 1. CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Char‑LSTM Shakespeare generator")
    p.add_argument("--seq_len", type=int, default=120, help="sequence length")
    p.add_argument("--batch", type=int, default=256, help="mini‑batch size")
    p.add_argument("--hidden", type=int, default=512, help="hidden units")
    p.add_argument("--embed", type=int, default=256, help="embedding size")
    p.add_argument("--layers", type=int, default=3, help="LSTM layers")
    p.add_argument("--dropout", type=float, default=0.3, help="dropout between LSTM layers")
    p.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    p.add_argument("--epochs", type=int, default=25, help="training epochs")
    p.add_argument("--clip", type=float, default=5.0, help="gradient clip")
    p.add_argument("--print_every", type=int, default=200, help="status interval")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="shakespeare.txt")
    p.add_argument("--model", default="char_lstm.pth", help="Model filename (saved in results/)")
    return p.parse_args()


# -----------------------------------------------------------------------------
# 2. Utils
# -----------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_PATH = os.path.join(RESULTS_DIR, "shakespeare.txt")
MODEL_PATH = os.path.join(RESULTS_DIR, "char_lstm.pth")  # 默认模型名
LOSS_CURVE_PATH = None  # 将在训练后动态生成

def download_shakespeare(path: str):
    if os.path.exists(path):
        return
    url = "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print("[✓] downloaded Shakespeare corpus →", path)
#

# -----------------------------------------------------------------------------
# 3. Dataset
# -----------------------------------------------------------------------------

def build_dataset(text: str, seq_len: int):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}

    sequences, targets = [], []
    for i in range(0, len(text) - seq_len):
        sequences.append([c2i[ch] for ch in text[i : i + seq_len]])
        targets.append(c2i[text[i + seq_len]])

    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return TensorDataset(X, y), c2i, i2c, len(chars)


# -----------------------------------------------------------------------------
# 4. Model
# -----------------------------------------------------------------------------


class CharLSTM(nn.Module):
    def __init__(self, vocab: int, embed_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)  # out: [B, T, H]
        logits = self.fc(out[:, -1, :])
        return logits, hidden


# -----------------------------------------------------------------------------
# 5. Training
# -----------------------------------------------------------------------------

def train(model, loader, criterion, optim, device, epochs, clip, print_every):
    loss_hist = []
    
    # Add progress bar for epochs
    epoch_bar = trange(1, epochs + 1, desc="Training", unit="epoch")
    
    for ep in epoch_bar:
        model.train()
        total, hidden = 0.0, None
        
        # Add progress bar for batches
        batch_bar = tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False, unit="batch")
        
        for step, (x, y) in enumerate(batch_bar, 1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            out, hidden = model(x, hidden)
            hidden = tuple(h.detach() for h in hidden)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()
            total += loss.item()
            
            # Update progress bar with current loss
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            if step % print_every == 0:
                # Keep original print but keep it minimal
                batch_bar.write(f"[ep {ep}] step {step}/{len(loader)} loss {loss.item():.4f}")
                
        avg = total / len(loader)
        loss_hist.append(avg)
        
        # Update epoch progress bar with mean loss
        epoch_bar.set_postfix(loss=f"{avg:.4f}")
        epoch_bar.write(f"→ epoch {ep} mean loss {avg:.4f}")
        
    return loss_hist


# -----------------------------------------------------------------------------
# 6. Sampling
# -----------------------------------------------------------------------------

def generate(model, seed: str, c2i, i2c, vocab: int, length=400, temp=1.0, device="cpu"):
    model.eval()
    out = seed
    seq = [c2i[c] for c in seed]
    hidden = None
    for _ in range(length):
        x = torch.tensor(seq[-args.seq_len :], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, hidden = model(x, hidden)
        probs = F.softmax(logits.squeeze() / temp, dim=0).cpu().numpy()
        idx = np.random.choice(vocab, p=probs)
        out += i2c[idx]
        seq.append(idx)
    return out


# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------


def main():
    global args  # needed inside generate()
    args = parse_args()
    set_seed(args.seed)
    download_shakespeare(args.data)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    ds, c2i, i2c, vocab = build_dataset(text, args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = CharLSTM(vocab, args.embed, args.hidden, args.layers, args.dropout).to(args.device)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = train(model, loader, crit, optim, args.device, args.epochs, args.clip, args.print_every)

    torch.save(model.state_dict(), MODEL_PATH)
    print("[✓] weights saved →", MODEL_PATH)

    LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, f"loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.figure()
    plt.plot(losses)
    plt.title("training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    print("[✓] curve saved →", LOSS_CURVE_PATH)
    

    for t in (0.5, 1.0, 1.3):
        print(f"\n--- temperature {t} ---")
        print(generate(model, "ROMEO: ", c2i, i2c, vocab, temp=t, device=args.device))

        sample_path = os.path.join(RESULTS_DIR, f"sample_temp_{t}.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(generate(model, "ROMEO: ", c2i, i2c, vocab, temp=t, device=args.device))
        print(f"[✓] sample saved → {sample_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
