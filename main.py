import time
import random
import numpy as np
import ollama
import csv

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import GSM8K, BigBenchHard, BoolQ

from methods.base import BaseModel
from methods.cot import CoTModel
from methods.tot import ToTModel
from methods.ssdp import SSDPModel
from methods.ototuss import OtotussModel

# ============================================================
# Reproducibility
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# Configuration
# ============================================================

TARGET_MODEL = "qwen2.5:7b"

NUM_PROBLEMS = 100
N_SHOTS = 0

# ============================================================
# Initialize Model
# ============================================================

base_model = BaseModel(model_name=TARGET_MODEL, seed=42)
cot_model = CoTModel(model_name=TARGET_MODEL, extract_final_answer=True, seed=42)
tot_model = ToTModel(model_name=TARGET_MODEL, search_strategy="bfs", extract_final_answer=True, seed=42)
ssdp_model = SSDPModel(model_name=TARGET_MODEL, extract_final_answer=True, seed=42)
ototuss_model = OtotussModel(model_name=TARGET_MODEL, extract_final_answer=True, seed=42)

models = [("Base Model", base_model), ("CoT Model", cot_model), ("ToT Model", tot_model), ("SSDP Model", ssdp_model), ("Ototuss Model", ototuss_model)]

# ============================================================
# Benchmark Runner
# ============================================================

def run_benchmark(name, benchmark, model_name, target_model):
    print("\n" + "=" * 70)
    print(f"Running {name} with {model_name}")
    print("=" * 70)

    start_tokens = target_model.token_usage
    start_time = time.time()

    benchmark.evaluate(model=target_model)

    elapsed = time.time() - start_time
    used_tokens = target_model.token_usage - start_tokens
    accuracy = benchmark.overall_score * 100

    print("\n" + "=" * 70)
    print(f"{name} Results for {model_name}")
    print("=" * 70)
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"Total Time    : {elapsed:.2f} sec")
    print(f"Avg Time      : {elapsed / NUM_PROBLEMS:.2f} sec/problem")
    print(f"Avg Tokens    : {used_tokens / NUM_PROBLEMS:.2f} tokens/problem")
    print("=" * 70)

    return {
        "Benchmark": name,
        "Method": model_name,
        "Accuracy": f"{accuracy:.2f}%",
        "Avg Time (s)": f"{elapsed / NUM_PROBLEMS:.2f}",
        "Avg Tokens": f"{used_tokens / NUM_PROBLEMS:.2f}"
    }


# ============================================================
# Benchmarks
# ============================================================

benchmark_classes = [
    ("GSM8K", GSM8K),
    ("BoolQ", BoolQ)
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
results = []

for name, benchmark_class in benchmark_classes:
    for model_name, target_model in models:
        print("\n" + "=" * 70)
        print(f"Evaluating {model_name} on {name}")
        print("=" * 70)
        
        # Instantiate fresh benchmark per model
        if name == "GSM8K":
            benchmark = benchmark_class(n_problems=NUM_PROBLEMS, n_shots=N_SHOTS, enable_cot=False)
        else:
            benchmark = benchmark_class(n_problems=NUM_PROBLEMS, n_shots=N_SHOTS)
            
        res = run_benchmark(name, benchmark, model_name, target_model)
        results.append(res)

overall_elapsed = time.time() - overall_start

csv_filename = "results.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Benchmark", "Method", "Accuracy", "Avg Time (s)", "Avg Tokens"])
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 70)
print("All Benchmarks Completed")
print("=" * 70)
print(f"Total Time  : {overall_elapsed:.2f} sec")
print(f"Total Tokens: {sum([m[1].token_usage for m in models])}")
print(f"Results saved to {csv_filename}")
print("=" * 70)