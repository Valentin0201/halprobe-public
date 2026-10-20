"""Environment check: GPU visibility, GPU-vs-CPU numerical agreement, model load, VRAM."""

from __future__ import annotations

import argparse
import platform
import time

import torch
import torch.nn.functional as F

from halprobe.device import resolve_device

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
QUESTIONS = (
    "Who was the director of Jaws?",
    "In what city was Marie Curie born?",
    "What is the capital of Burkina Faso?",
)


def kernel_checks(device: torch.device) -> bool:
    """Compare fp16 kernels on `device` against an fp32 CPU reference."""
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, dtype=torch.float16)
    b = torch.randn(1024, 1024, dtype=torch.float16)
    h = torch.randn(4, 256, 1024, dtype=torch.float16)
    w = torch.ones(1024, dtype=torch.float16)
    x = torch.randn(2, 8, 64, 64, dtype=torch.float16)
    checks = {
        "matmul": (torch.matmul, (a, b)),
        "softmax": (lambda t: torch.softmax(t, -1), (h,)),
        "layer_norm": (lambda t, g: F.layer_norm(t, (1024,), g), (h, w)),
        "sdpa_causal": (lambda q: F.scaled_dot_product_attention(q, q, q, is_causal=True), (x,)),
        "silu": (F.silu, (h,)),
    }
    ok = True
    for name, (fn, args) in checks.items():
        reference = fn(*[t.float() for t in args])
        result = fn(*[t.to(device) for t in args]).float().cpu()
        err = ((result - reference).abs().max() / reference.abs().max().clamp_min(1e-6)).item()
        passed = err < 2e-2
        ok = ok and passed
        print(f"  {name:<12} rel_err={err:.2e} {'OK' if passed else 'FAIL'}")
    return ok


def gpu_report(device: torch.device) -> dict:
    """Matmul throughput and free/total VRAM on `device`."""
    a = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - start) / 20
    free, total = torch.cuda.mem_get_info(device)
    return {"tflops_fp16": 2 * 2048**3 / seconds / 1e12, "vram_free_gib": free / 2**30, "vram_total_gib": total / 2**30}


@torch.no_grad()
def model_check(model_id: str, revision: str, device: torch.device) -> bool:
    """Load the model in fp16 on `device` and fp32 on CPU; compare hidden states and greedy answers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024"
        )
        for q in QUESTIONS
    ]
    torch.cuda.reset_peak_memory_stats(device)
    gpu = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float16).to(device).eval()
    cpu = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32).eval()
    ok = True
    for question, prompt in zip(QUESTIONS, prompts):
        results = {}
        for name, model, dev in (("gpu", gpu, device), ("cpu", cpu, torch.device("cpu"))):
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
            out = model(**enc, output_hidden_states=True)
            states = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
            gen = model.generate(
                **enc, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id
            )
            answer = tokenizer.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            results[name] = (states, answer)
        cos = F.cosine_similarity(results["gpu"][0][1:], results["cpu"][0][1:], dim=-1).min().item()
        same = results["gpu"][1] == results["cpu"][1]
        ok = ok and cos > 0.99
        print(f"  {question}\n    gpu={results['gpu'][1]!r} cpu={results['cpu'][1]!r} "
              f"{'same' if same else 'DIFFERENT (warning)'} | min hidden cosine {cos:.4f}")
    print(f"  peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    return ok


def tiny_cpu_check() -> bool:
    """Forward pass of a tiny random Llama on CPU (no downloads)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 128, (2, 10)), output_hidden_states=True)
    ok = out.logits.shape == (2, 10, 128) and len(out.hidden_states) == 3 and bool(torch.isfinite(out.logits).all())
    print(f"  tiny CPU forward pass: {'OK' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the GPU/ROCm setup for halprobe.")
    parser.add_argument("--cpu", action="store_true", help="CPU-only check with a tiny random model")
    parser.add_argument("--gpu-name-hint", default="6700")
    parser.add_argument("--skip-model", action="store_true", help="skip loading the real model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args(argv)

    print(f"python {platform.python_version()} | torch {torch.__version__} | hip {torch.version.hip} "
          f"| GPU available {torch.cuda.is_available()}")
    if args.cpu:
        ok = tiny_cpu_check()
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [cuda:{i}] {props.name} | {getattr(props, 'gcnArchName', 'n/a')} | {props.total_memory / 2**30:.2f} GiB")
        device = resolve_device("auto", args.gpu_name_hint)
        print(f"selected {device} ({torch.cuda.get_device_name(device)})")
        ok = kernel_checks(device)
        info = gpu_report(device)
        print(f"  matmul {info['tflops_fp16']:.1f} TFLOP/s | VRAM free {info['vram_free_gib']:.2f} / {info['vram_total_gib']:.2f} GiB")
        if not args.skip_model:
            ok = model_check(args.model, args.revision, device) and ok
    print("ENV CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1
