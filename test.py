"""Host check: C runtime (W4A8) vs the PyTorch fp32 forward of the same checkpoint.

python test.py            # downloads the QAT-nf4 checkpoint if missing, exports, builds, compares
"""
import os, subprocess, sys, urllib.request
import numpy as np
import torch
from flashee.model import load_checkpoint

CKPT, BIN, EXE = "flashee-qat-nf4.pth", "model.bin", "./flashee_host"
SEQS = ["MKT", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "ACDEFGHIKLMNPQRSTVWYXBZU" * 6]                                          # 3, 76 (ubiquitin), 144 residues

if not os.path.exists(CKPT):
    urllib.request.urlretrieve("https://huggingface.co/szchesny/flashee/resolve/main/" + CKPT, CKPT)
subprocess.run([sys.executable, "export.py", CKPT, BIN], check=True)
subprocess.run(["cc", "-O2", "-o", EXE, "host.c", "flashee.c", "-lm"], check=True)

model, vocab = load_checkpoint(CKPT)
model.eval()
worst = 0
for seq in SEQS:
    ids = [vocab.cls] + [vocab.stoi.get(a, vocab.unk) for a in seq] + [vocab.eos]
    with torch.no_grad():
        ref = model(torch.tensor([ids]), return_repr=True)[0, 1:-1].numpy()
    got = np.array([[float(v) for v in line.split()] for line in
                    subprocess.check_output([EXE, BIN, seq, "--per-residue"]).decode().splitlines()])
    assert got.shape == ref.shape, (got.shape, ref.shape)
    cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
    err = np.abs(got - ref).max() / np.abs(ref).max()
    worst = max(worst, err)
    print(f"L={len(seq):4d}  min cos {cos.min():.5f}  max |err| {err:.4f} of max |ref|")
    assert cos.min() > 0.995, "C forward disagrees with PyTorch"
print(f"OK: C runtime matches fp32 PyTorch to {worst:.3%} (residual is int8 activations, by design)")
