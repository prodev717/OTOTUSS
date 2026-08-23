import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import time
import csv
import random
import numpy as np
from deepeval.test_case import LLMTestCase

from methods.ototuss import OtotussModel
from custom_metrics import relevancy_metric, creativity_metric, coherence_metric, emotion_metric, quality_metric

def main():
    # Setup reproducibility
    random.seed(43)
    np.random.seed(43)

    TARGET_MODEL = "qwen2.5:7b"
    
    prompts = [
        "Write a short, emotional poem about a time traveler who falls in love with a Roman gladiator.",
        "Write a thrilling short story (under 200 words) about a robot discovering a forgotten magical library.",
        "Compose a creative marketing pitch for a fictional device that can record and playback dreams."
    ]
    
    metrics = {
        "Relevancy": relevancy_metric,
        "Creativity": creativity_metric,
        "Coherence": coherence_metric,
        "Emotion": emotion_metric,
        "Quality": quality_metric
    }
    
    semantic_thresholds = [0.2, 0.5, 0.8]
    termination_thresholds = [8.0, 9.0, 9.5]
    
    results = []
    
    print("=" * 70)
    print("Starting Grid Search for OtotussModel (Creative Writing)")
    print(f"Model: {TARGET_MODEL}, Prompts: {len(prompts)}")
    print("=" * 70)
    
    print("\nInitializing model and embedding models...")
    # creative writing typically needs detailed output, so extract_final_answer=False
    model = OtotussModel(model_name=TARGET_MODEL, extract_final_answer=False, seed=43)
    
    overall_start = time.time()
    
    for sem_thresh in semantic_thresholds:
        for term_thresh in termination_thresholds:
            print(f"\nTesting semantic_threshold={sem_thresh}, early_termination_threshold={term_thresh}")
            
            model.semantic_threshold = sem_thresh
            model.early_termination_threshold = term_thresh
            
            total_time = 0
            total_tokens = 0
            total_score = 0
            
            for i, prompt in enumerate(prompts):
                model.token_usage = 0
                start_time = time.time()
                
                actual_output = model.generate(prompt)
                
                elapsed = time.time() - start_time
                used_tokens = model.token_usage
                
                total_time += elapsed
                total_tokens += used_tokens
                
                test_case = LLMTestCase(
                    input=prompt,
                    actual_output=actual_output,
                    retrieval_context=[prompt] 
                )
                
                prompt_score = 0
                for metric_name, metric in metrics.items():
                    try:
                        metric.measure(test_case)
                        prompt_score += metric.score
                    except Exception:
                        pass
                
                # Average score for this prompt (out of 5 metrics, each 0-1 range usually, or whatever metric.score is)
                # Assuming score is 0.0 to 1.0, prompt_score / len(metrics) is the average score for this prompt
                avg_prompt_score = prompt_score / len(metrics)
                total_score += avg_prompt_score
                
            # Average score across all prompts (scaled to 100%)
            overall_accuracy = (total_score / len(prompts)) * 100
            
            print(f"Creative Score: {overall_accuracy:.2f}% | Total Time: {total_time:.2f}s | Total Tokens: {total_tokens}")
            
            results.append({
                "Semantic Threshold": sem_thresh,
                "Termination Threshold": term_thresh,
                "Creative Score (%)": f"{overall_accuracy:.2f}",
                "Avg Time (s)": f"{total_time / len(prompts):.2f}",
                "Avg Tokens": f"{total_tokens / len(prompts):.2f}",
                "Total Tokens": total_tokens
            })
            
    overall_elapsed = time.time() - overall_start
    
    csv_filename = "ototuss_creative_tuning_results.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Semantic Threshold", 
            "Termination Threshold", 
            "Creative Score (%)", 
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
