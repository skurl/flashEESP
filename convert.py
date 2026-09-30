"""fp32 flashee .pth -> int8 .tflite for TensorFlow Lite Micro.

.venv/bin/python convert.py flashee.pth calib.fasta --len 512 --out flashee_int8.tflite

Static shapes: tokens int32 [1, LEN] (<cls> seq <eos> then <pad>), mask float [1, LEN] (1 real, 0 pad).
Output: per-token representation float [1, LEN, 320]; pool over the real rows on the device.

Recipes (--recipe):
  fc    int8 FULLY_CONNECTED (weights per-channel, activations per-tensor), everything else float.  default
        This is the only int8 the ESP32-S3 accelerates (esp-nn), and it keeps the LayerNorm/GELU/residual chain exact.
  full  int8 everywhere (default_a8w8).  Measured: collapses the representation (mean cos 0.7).  kept for the record
  a16   int16 activations everywhere.  Better (mean cos 0.997) but esp-nn does not accelerate int16.
Attention is computed --qchunk query rows at a time: exact (softmax is per query row) and the planner reuses one
[H, qchunk, L] score buffer, so memory grows ~linearly in L instead of H*L*L floats (8 MB at L=512 = all of PSRAM).
SmoothQuant (--alpha, 0 disables) folds per-channel activation outliers into the preceding LayerNorm / W_v.
"""
import argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer, recipe, qtyping

MASK_BIAS = -16.0          # additive logit for pad keys: exp(-16) ~ 1e-7


def read_fasta(path):
    seq = []
    for line in open(path):
        if line.startswith(">"):
            if seq: yield "".join(seq)
            seq = []
        else:
            seq.append(line.strip())
    if seq: yield "".join(seq)


class Encoder(nn.Module):
    """flashee with ops TFLM has: no SDPA, no erf-GELU, LayerNorm spelled out, RoPE tables as constants."""

    def __init__(self, sd, arch, L, qchunk=0):
        super().__init__()
        self.qchunk = qchunk if 0 < qchunk < L else L
        d, self.H, self.nl = arch["d_model"], arch["num_heads"], arch["num_layers"]
        hd = d // self.H
        self.embed = nn.Embedding.from_pretrained(sd["embed.weight"], freeze=True)
        self.p = nn.ParameterDict({k.replace(".", "_"): nn.Parameter(v.clone(), requires_grad=False) for k, v in sd.items()})
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        fr = torch.outer(torch.arange(L).float(), inv)                       # same float32 recipe as the reference
        self.register_buffer("cos", fr.cos()[None, None]); self.register_buffer("sin", fr.sin()[None, None])
        self.scale = 1.0 / math.sqrt(hd)
        self.stats = None                                                     # dict while collecting SmoothQuant ranges

    def ln(self, x, w, b):
        mu = x.mean(-1, keepdim=True); xc = x - mu
        return xc * torch.rsqrt((xc * xc).mean(-1, keepdim=True) + 1e-5) * w + b

    def rope(self, t):
        a, b = t.chunk(2, dim=-1)
        return torch.cat([a * self.cos - b * self.sin, b * self.cos + a * self.sin], -1)

    def rec(self, key, x, mask):
        if self.stats is not None:
            self.stats.setdefault(key, []).append(x[0][mask[0] > 0].abs().amax(0))

    def forward(self, tok, mask):
        p, H = self.p, self.H
        B, L = tok.shape
        x = self.embed(tok)
        bias = ((mask - 1.0) * -MASK_BIAS).reshape(B, 1, 1, L)
        for i in range(self.nl):
            g = lambda n: p[f"layers_{i}_{n}"]
            h = self.ln(x, g("norm1_weight"), g("norm1_bias")); self.rec(f"{i}.h1", h, mask)
            split = lambda t: t.reshape(B, L, H, -1).permute(0, 2, 1, 3)
            q, k, v = (split(F.linear(h, g(f"attn_W_{n}_weight"))) for n in "qkv")
            qr, kt = self.rope(q), self.rope(k).transpose(-1, -2)
            attn = lambda qc: torch.matmul(torch.softmax(torch.matmul(qc, kt) * self.scale + bias, -1), v)
            o = torch.cat([attn(qr[:, :, c:c + self.qchunk]) for c in range(0, L, self.qchunk)], 2)
            o = o.permute(0, 2, 1, 3).reshape(B, L, -1); self.rec(f"{i}.o", o, mask)
            x = x + F.linear(o, g("attn_W_o_weight"))
            h = self.ln(x, g("norm2_weight"), g("norm2_bias")); self.rec(f"{i}.h2", h, mask)
            f = F.linear(h, g("ff_0_weight"), g("ff_0_bias"))
            f = f * torch.sigmoid(1.5957691 * (f + 0.044715 * f * f * f))  # tanh-GELU via LOGISTIC: TFLM has no GELU op
            x = x + F.linear(f, g("ff_2_weight"), g("ff_2_bias"))
        return self.ln(x, p["final_norm_weight"], p["final_norm_bias"])

    @torch.no_grad()
    def smooth(self, batches, alpha):
        """SmoothQuant: x_j / s_j into the LayerNorm (or W_v row), W[:, j] * s_j into the consumer. Exact in float."""
        self.stats = {}
        for t, m in batches: self(t, m)
        A = {k: torch.stack(v).amax(0) for k, v in self.stats.items()}; self.stats = None
        sf = lambda a, wmax: (a.clamp(min=1e-5) ** alpha / wmax.clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)
        for i in range(self.nl):
            g = lambda n: self.p[f"layers_{i}_{n}"]
            s = sf(A[f"{i}.h1"], torch.stack([g(f"attn_W_{n}_weight").abs().amax(0) for n in "qkv"]).amax(0))
            g("norm1_weight").div_(s); g("norm1_bias").div_(s)
            for n in "qkv": g(f"attn_W_{n}_weight").mul_(s)
            s = sf(A[f"{i}.h2"], g("ff_0_weight").abs().amax(0))
            g("norm2_weight").div_(s); g("norm2_bias").div_(s); g("ff_0_weight").mul_(s)
            s = sf(A[f"{i}.o"], g("attn_W_o_weight").abs().amax(0))            # attention mixes tokens, not channels
            g("attn_W_v_weight").div_(s[:, None]); g("attn_W_o_weight").mul_(s)


def encode(seq, vocab, L):
    stoi = {s: i for i, s in enumerate(vocab)}
    ids = [stoi["<cls>"]] + [stoi.get(a, stoi["<unk>"]) for a in seq.upper()[:L - 2]] + [stoi["<eos>"]]
    tok = np.zeros((1, L), np.int32); tok[0, :len(ids)] = ids
    mask = np.zeros((1, L), np.float32); mask[0, :len(ids)] = 1
    return tok, mask, len(ids) - 2


def run_tflite(path, tok, mask):
    it = Interpreter(model_path=path); it.allocate_tensors()
    for d in it.get_input_details():
        x = tok if d["dtype"] == np.int32 else mask
        if d["dtype"] in (np.int8, np.int16):                    # the quantizer makes float IO integer
            sc, zp = d["quantization"]; lo = np.iinfo(d["dtype"]).min
            x = np.clip(np.round(x / sc) + zp, lo, -lo - 1).astype(d["dtype"])
        it.set_tensor(d["index"], x)
    it.invoke()
    d = it.get_output_details()[0]; y = it.get_tensor(d["index"])[0]
    if d["dtype"] in (np.int8, np.int16):
        sc, zp = d["quantization"]; y = (y.astype(np.float32) - zp) * sc
    return y


def fix_for_tflm(path):
    """Two rewrites TFLM needs, both exact:
    1. Buffers stored after the flatbuffer via offset/size (what the converter/quantizer emit) are invisible to
       esp-tflite-micro: it silently reads empty weights. Move every buffer back inside the flatbuffer.
    2. TFLM's BATCH_MATMUL keeps a *persistent* transposed copy of its RHS unless adj_y is set: 655 KB per
       matmul at L=512, x96 with query chunking. Give every BATCH_MATMUL an RHS it can use as-is: K^T comes from a
       TRANSPOSE, so feed K itself; V gets one explicit TRANSPOSE per layer (a normal, reused activation)."""
    import flatbuffers
    from ai_edge_litert import schema_py_generated as S
    raw = open(path, "rb").read()
    m = S.ModelT.InitFromPackedBuf(raw, 0)
    for b in m.buffers:
        if b.offset > 1:                     # offset 1 is the "no data" sentinel
            b.data = np.frombuffer(raw, np.uint8, b.size, b.offset)
        b.offset = b.size = 0

    g = m.subgraphs[0]
    code = lambda op: max(m.operatorCodes[op.opcodeIndex].builtinCode, m.operatorCodes[op.opcodeIndex].deprecatedBuiltinCode)
    BMM, TR = S.BuiltinOperator.BATCH_MATMUL, S.BuiltinOperator.TRANSPOSE
    producer = {t: op for op in g.operators for t in op.outputs}
    perm = lambda op: list(np.frombuffer(bytes(m.buffers[g.tensors[op.inputs[1]].buffer].data), np.int32))
    swaps_last2 = lambda op: code(op) == TR and perm(op) == list(range(len(perm(op)) - 2)) + [len(perm(op)) - 1, len(perm(op)) - 2]
    tr_code = next(op.opcodeIndex for op in g.operators if code(op) == TR)
    vT, new_ops = {}, []
    for op in g.operators:
        if code(op) == BMM and not op.builtinOptions.adjY:
            rhs = op.inputs[1]
            if rhs in producer and swaps_last2(producer[rhs]):
                op.inputs = [op.inputs[0], producer[rhs].inputs[0]] + list(op.inputs[2:])  # K^T = transpose(K): use K
            else:                                                 # V: transpose once, shared by every query chunk
                if rhs not in vT:
                    t = g.tensors[rhs]; n = len(t.shape)
                    pb = S.BufferT(); pb.data = np.array(list(range(n - 2)) + [n - 1, n - 2], np.int32).view(np.uint8)
                    m.buffers.append(pb)
                    pt = S.TensorT(); pt.shape = [n]; pt.type = S.TensorType.INT32; pt.buffer = len(m.buffers) - 1; pt.name = b"vT_perm"
                    ot = S.TensorT(); ot.shape = list(t.shape[:-2]) + [t.shape[-1], t.shape[-2]]; ot.type = t.type; ot.buffer = 0; ot.name = t.name + b"_T"
                    g.tensors += [pt, ot]
                    top = S.OperatorT(); top.opcodeIndex = tr_code; top.inputs = [rhs, len(g.tensors) - 2]; top.outputs = [len(g.tensors) - 1]
                    top.builtinOptionsType = S.BuiltinOptions.TransposeOptions; top.builtinOptions = S.TransposeOptionsT()
                    new_ops.append(top); vT[rhs] = len(g.tensors) - 1
                op.inputs = [op.inputs[0], vT[rhs]] + list(op.inputs[2:])
            op.builtinOptions.adjY = True
        new_ops.append(op)
    used = lambda ops: {t for op in ops for t in op.inputs} | set(g.outputs)
    while True:                                                   # drop TRANSPOSEs nobody reads any more
        u = used(new_ops); keep = [op for op in new_ops if any(t in u for t in op.outputs)]
        if len(keep) == len(new_ops): break
        new_ops = keep
    g.operators = new_ops
    fb = flatbuffers.Builder(len(raw) + (1 << 20))
    fb.Finish(m.Pack(fb), file_identifier=b"TFL3")
    open(path, "wb").write(fb.Output())


def tflite_ops(path):
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils as u
    from ai_edge_litert import schema_py_generated as schema
    m = u.read_model(path)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    return sorted({names[m.operatorCodes[op.opcodeIndex].builtinCode] for op in m.subgraphs[0].operators})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("calib_fasta")
    ap.add_argument("--len", type=int, default=512, help="tokens incl. <cls>/<eos>; residues = len-2 (trained at 512)")
    ap.add_argument("--qchunk", type=int, default=64, help="query rows per attention block, 0 = whole L x L map")
    ap.add_argument("--out", default="flashee_int8.tflite")
    ap.add_argument("--recipe", default="fc", choices=["fc", "full", "a16"])
    ap.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant strength, 0 = off")
    ap.add_argument("--n_eval", type=int, default=20)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd, arch, vocab, L = ck["model"], ck["arch"], ck["vocab"], a.len
    model = Encoder(sd, arch, L, a.qchunk).eval()
    seqs = [s for s in read_fasta(a.calib_fasta) if len(s) <= L - 2]
    calib, evals = seqs[:-a.n_eval], seqs[-a.n_eval:]
    enc = [encode(s, vocab, L) for s in calib]
    print(f"{len(calib)} calibration / {len(evals)} eval sequences, L={L}")
    if a.alpha > 0:
        model.smooth([(torch.from_numpy(t), torch.from_numpy(m)) for t, m, _ in enc[:80]], a.alpha)

    t0, m0, _ = enc[0]
    litert_torch.convert(model, (torch.from_numpy(t0), torch.from_numpy(m0))).export("flashee_float.tflite")

    from flashee.model import load_checkpoint                    # the unmodified reference forward
    ref_model, rv = load_checkpoint(a.ckpt); ref_model.eval()

    def reference(seq):
        ids = [rv.cls] + [rv.stoi.get(c, rv.unk) for c in seq.upper()] + [rv.eos]
        with torch.no_grad():
            return ref_model(torch.tensor([ids]), return_repr=True)[0, 1:-1].numpy()

    def score(path, name):
        cs = []
        for s in evals:
            t, m, n = encode(s, vocab, L)
            got, ref = run_tflite(path, t, m)[1:n + 1], reference(s)
            cs.append((got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1)))
        cs = np.concatenate(cs)
        print(f"{name:12s} vs fp32 PyTorch, per-residue cosine: min {cs.min():.4f}  1st pct {np.percentile(cs, 1):.4f}  mean {cs.mean():.4f}")
        return cs.mean(), cs.min()

    score("flashee_float.tflite", "float tflite")

    it = Interpreter(model_path="flashee_float.tflite")
    sig = list(it.get_signature_list())[0]
    names = it.get_signature_list()[sig]["inputs"]
    data = [{names[0]: t, names[1]: m} for t, m, _ in enc]
    if a.recipe == "fc":
        T = qtyping.TensorQuantizationConfig
        cfg = qtyping.OpQuantizationConfig(
            activation_tensor_config=T(num_bits=8, symmetric=False, granularity=qtyping.QuantGranularity.TENSORWISE),
            weight_tensor_config=T(num_bits=8, symmetric=True, granularity=qtyping.QuantGranularity.CHANNELWISE),
            compute_precision=qtyping.ComputePrecision.INTEGER)
        qt = quantizer.Quantizer("flashee_float.tflite")
        qt.update_quantization_recipe(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, op_config=cfg)
    else:
        qt = quantizer.Quantizer("flashee_float.tflite", recipe.static_wi8_ai8() if a.recipe == "full" else recipe.static_wi8_ai16())
    res = qt.quantize(qt.calibrate({sig: data}), serialize_to_path=a.out, enable_progress_report=False)
    fix_for_tflm(a.out)
    print(f"{a.out}: {len(res.quantized_model) / 2**20:.2f} MiB   ops: {' '.join(tflite_ops(a.out))}")
    mean, mn = score(a.out, f"int8 ({a.recipe})")
    assert mean > 0.98 and mn > 0.85, "quantised model lost the representation"


if __name__ == "__main__":
    main()
