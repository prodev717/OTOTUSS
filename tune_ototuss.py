import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import time
import csv
import random
import numpy as np
from deepeval.benchmarks import GSM8K
from methods.ototuss import OtotussModel

def main():
    # Setup reproducibility
    random.seed(43)
    np.random.seed(43)
    
    TARGET_MODEL = "qwen2.5:7b"
    NUM_PROBLEMS = 10
    N_SHOTS = 0
    
    semantic_thresholds = [0.2, 0.5, 0.8]
    termination_thresholds = [8.0, 9.0, 9.5]
    
    results = []
    
    # Initialize benchmark inside the loop to avoid caching
    
    print("=" * 70)
    print("Starting Grid Search for OtotussModel")
    print(f"Model: {TARGET_MODEL}, Problems: {NUM_PROBLEMS}")
    print("=" * 70)
    
    # Initialize model ONCE to avoid reloading the embedding model every iteration
    print("\nInitializing model and embedding models...")
    model = OtotussModel(model_name=TARGET_MODEL, extract_final_answer=True, seed=43)
    
    overall_start = time.time()
    
    for sem_thresh in semantic_thresholds:
        for term_thresh in termination_thresholds:
            print(f"\nTesting semantic_threshold={sem_thresh}, early_termination_threshold={term_thresh}")
            
            # Instantiate fresh benchmark per run
            gsm8k = GSM8K(n_problems=NUM_PROBLEMS, n_shots=N_SHOTS, enable_cot=False)
            
            # Update thresholds on the existing model
            model.semantic_threshold = sem_thresh
            model.early_termination_threshold = term_thresh
            model.token_usage = 0  # Reset token usage for this run
            
            start_tokens = model.token_usage
            start_time = time.time()
            
            # Run benchmark
            gsm8k.evaluate(model=model)
            
            elapsed = time.time() - start_time
            used_tokens = model.token_usage - start_tokens
            accuracy = gsm8k.overall_score * 100
            
            print(f"Accuracy: {accuracy:.2f}% | Time: {elapsed:.2f}s | Tokens: {used_tokens}")
            
            results.append({
                "Semantic Threshold": sem_thresh,
                "Termination Threshold": term_thresh,
                "Accuracy (%)": f"{accuracy:.2f}",
                "Avg Time (s)": f"{elapsed / NUM_PROBLEMS:.2f}",
                "Avg Tokens": f"{used_tokens / NUM_PROBLEMS:.2f}",
                "Total Tokens": used_tokens
            })
            
    overall_elapsed = time.time() - overall_start
    
    csv_filename = "ototuss_tuning_results.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Semantic Threshold", 
            "Termination Threshold", 
            "Accuracy (%)", 
            "Avg Time (s)", 
            "Avg Tokens",
            "Total Tokens"
        ])
        writer.writeheader()
        writer.writerows(results)
        
    print("\n" * 2 + "=" * 70)
    print("Tuning Completed")
    print("=" * 70)
    print(f"Total Tuning Time: {overall_elapsed:.2f} sec")
    print(f"Results saved to {csv_filename}")
    print("=" * 70)

if __name__ == "__main__":
    main()
