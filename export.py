"""flashee .pth -> model.bin for the C runtime.

python export.py flashee-qat-nf4.pth model.bin

Layout (little-endian, every section a multiple of 4 bytes):
  header   'FLEE' u32, then u32 d_model n_heads n_layers d_ff vocab n_classes, u8 tok[32] ('A'..'Z' -> id)
  embed    f32[vocab][d]
  layer x6 ln1_w ln1_b | Wq Wk Wv Wo | ln2_w ln2_b | W1 b1 | W2 b2
  final_ln w b
  fc       nf4 [n_classes][d]
nf4 matrix = f32 scale[out] then u8 nibbles[out][in/2], low nibble = even column, index into NF4 table.
"""
import struct, sys
import numpy as np
import torch

NF4 = torch.tensor([-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0,
                    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0])


def nf4(w):
    """per-out-channel absmax scale, snap to the NF4 codebook. -> bytes. Asserts the round trip is exact
    for a checkpoint that is already baked onto this grid (the QAT-nf4 one is)."""
    s = w.abs().amax(1, keepdim=True)
    idx = ((w / s.clamp(min=1e-12))[:, :, None] - NF4).abs().argmin(-1)
    assert (NF4[idx] * s - w).abs().max() < 1e-6, "weight is not on the per-channel nf4 grid"
    q = idx.to(torch.uint8).numpy()
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return s.squeeze(1).numpy().astype("<f4").tobytes() + packed.tobytes()


def f32(t):
    return t.numpy().astype("<f4").tobytes()


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    a, sd, vocab = ck["arch"], ck["model"], ck["vocab"]
    stoi = {s: i for i, s in enumerate(vocab)}
    tok = bytes(stoi.get(chr(c), stoi["<unk>"]) for c in range(ord("A"), ord("Z") + 1)) + bytes(6)
    assert [stoi[s] for s in ("<pad>", "<cls>", "<eos>")] == [0, 2, 4], "C runtime hardcodes these ids"
    out = [b"FLEE", struct.pack("<6I", a["d_model"], a["num_heads"], a["num_layers"], a["d_ff"],
                                a["vocab_size"], a["num_classes"]), tok, f32(sd["embed.weight"])]
    for i in range(a["num_layers"]):
        p = f"layers.{i}."
        out += [f32(sd[p + "norm1.weight"]), f32(sd[p + "norm1.bias"])]
        out += [nf4(sd[p + f"attn.W_{n}.weight"]) for n in "qkvo"]
        out += [f32(sd[p + "norm2.weight"]), f32(sd[p + "norm2.bias"]),
                nf4(sd[p + "ff.0.weight"]), f32(sd[p + "ff.0.bias"]),
                nf4(sd[p + "ff.2.weight"]), f32(sd[p + "ff.2.bias"])]
    out += [f32(sd["final_norm.weight"]), f32(sd["final_norm.bias"]), nf4(sd["fc.weight"])]
    blob = b"".join(out)
    assert len(blob) % 4 == 0
    open(dst, "wb").write(blob)
    print(f"{dst}: {len(blob) / 2**20:.2f} MiB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
