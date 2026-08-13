import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import time
import csv
from deepeval.test_case import LLMTestCase

from methods.ssdp import SSDPModel
from methods.ototuss import OtotussModel
from methods.base import BaseModel
from custom_metrics import relevancy_metric, creativity_metric, coherence_metric, emotion_metric, quality_metric

import random
import numpy as np

random.seed(42)
np.random.seed(42)


def main():
    TARGET_MODEL = "qwen2.5:7b"
    
    print(f"Initializing generation models ({TARGET_MODEL})...")
    ssdp_model = SSDPModel(model_name=TARGET_MODEL, extract_final_answer=False, seed=42)
    ototuss_model = OtotussModel(model_name=TARGET_MODEL, extract_final_answer=False, seed=42)
    
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
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print("\n" + "="*70)
        print(f"Prompt {i+1}: {prompt}")
        print("="*70)
        
        for model_name, model in [("SSDP", ssdp_model), ("Ototuss", ototuss_model)]:
            print(f"\n--- Generating with {model_name} ---")
            
            # Reset token usage for the model
            model.token_usage = 0
            
            start_time = time.time()
            
            # Generate the creative text
            actual_output = model.generate(prompt)
            
            elapsed_time = time.time() - start_time
            tokens_used = model.token_usage
            
            snippet = actual_output.replace('\n', ' ')
            print(f"Output generated in {elapsed_time:.2f}s. Tokens used: {tokens_used}")
            print(f"Output snippet: {snippet[:100]}...\n")
            
            # Create a test case
            test_case = LLMTestCase(
                input=prompt,
                actual_output=actual_output,
                retrieval_context=[prompt] 
            )
            
            result_row = {
                "Prompt ID": i+1,
                "Prompt": prompt,
                "Model": model_name,
                "Generation Time (s)": f"{elapsed_time:.2f}",
                "Tokens Used": tokens_used,
                "Actual Output": actual_output
            }
            
            for metric_name, metric in metrics.items():
                print(f"Evaluating {metric_name}...")
                try:
                    metric.measure(test_case)
                    score = metric.score
                    reason = metric.reason
                except Exception as e:
                    print(f"Evaluation failed for {metric_name}: {e}")
                    score = 0.0
                    reason = str(e)
                
                print(f"{metric_name} Score: {score}")
                print(f"{metric_name} Reason: {reason}")
                
                result_row[f"{metric_name} Score"] = score
                result_row[f"{metric_name} Reason"] = reason
            
            results.append(result_row)
            
    csv_filename = "creative_writing_results.csv"
    
    fieldnames = [
        "Prompt ID", "Prompt", "Model", "Generation Time (s)", "Tokens Used"
    ]
    for metric_name in metrics.keys():
        fieldnames.extend([f"{metric_name} Score", f"{metric_name} Reason"])
    fieldnames.append("Actual Output")

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("\n" + "="*70)
    print(f"Creative writing evaluation completed! Results saved to {csv_filename}")
    print("="*70)

if __name__ == "__main__":
    main()
