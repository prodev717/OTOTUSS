import time
import random
import numpy as np
import ollama

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import GSM8K, BigBenchHard, BoolQ

from methods.cot import CoTModel
from methods.base import BaseModel

# ============================================================
# Reproducibility
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# Configuration
# ============================================================

TARGET_MODEL = "qwen2.5:7b"

NUM_PROBLEMS = 10
N_SHOTS = 0

# ============================================================
# Initialize Model
# ============================================================

base_model = BaseModel(model_name=TARGET_MODEL)
cot_model = CoTModel(model_name=TARGET_MODEL)
models = [("Base Model", base_model), ("CoT Model", cot_model)]

# ============================================================
# Benchmark Runner
# ============================================================

def run_benchmark(name, benchmark):
    print("\n" + "=" * 70)
    print(f"Running {name}")
    print("=" * 70)

    start_tokens = target_model.token_usage
    start_time = time.time()

    benchmark.evaluate(model=target_model)

    elapsed = time.time() - start_time
    used_tokens = target_model.token_usage - start_tokens

    print("\n" + "=" * 70)
    print(f"{name} Results")
    print("=" * 70)
    print(f"Accuracy      : {benchmark.overall_score * 100:.2f}%")
    print(f"Total Time    : {elapsed:.2f} sec")
    print(f"Avg Time      : {elapsed / NUM_PROBLEMS:.2f} sec/problem")
    print(f"Avg Tokens    : {used_tokens / NUM_PROBLEMS:.2f} tokens/problem")
    print("=" * 70)


# ============================================================
# Benchmarks
# ============================================================

gsm8k = GSM8K(n_problems=NUM_PROBLEMS, n_shots=N_SHOTS, enable_cot=False)
# bbh = BigBenchHard(n_shots=N_SHOTS, enable_cot=False)
boolq = BoolQ(n_problems=NUM_PROBLEMS, n_shots=N_SHOTS)

# benchmarks = [
#     ("GSM8K", gsm8k),
#     ("BBH", bbh),
#     ("BoolQ", boolq)
# ]
benchmarks = [
    ("GSM8K", gsm8k),
    ("BoolQ", boolq)
]

# ============================================================
# Run All Benchmarks
# ============================================================

print("=" * 70)
print("DeepEval Benchmark Suite")
print("=" * 70)
print(f"Model       : {TARGET_MODEL}")
print(f"Problems    : {NUM_PROBLEMS}")
print(f"Few-shot    : {N_SHOTS}")
print("=" * 70)

overall_start = time.time()

for name, benchmark in benchmarks:
    for model_name, target_model in models:
        print("\n" + "=" * 70)
        print(f"Evaluating {model_name} on {name}")
        print("=" * 70)
        run_benchmark(name, benchmark)

overall_elapsed = time.time() - overall_start

print("\n" + "=" * 70)
print("All Benchmarks Completed")
print("=" * 70)
print(f"Total Time  : {overall_elapsed:.2f} sec")
print(f"Total Tokens: {target_model.token_usage}")
print("=" * 70)