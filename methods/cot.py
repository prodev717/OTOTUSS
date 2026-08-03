import ollama
from deepeval.models.base_model import DeepEvalBaseLLM


class CoTModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str, extract_final_answer: bool = True):
        self.model_name = model_name
        self.extract_final_answer = extract_final_answer
        self.token_usage = 0

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema=None, **kwargs):
        cot_prompt = f"{prompt} \nWork through the problem step by step before giving the final answer."

        try:
            # Step 1: Generate reasoning
            res1 = ollama.generate(
                model=self.model_name,
                prompt=cot_prompt,
                options={
                    "temperature": 0,
                    "seed": 42,
                },
            )
            reasoning = res1["response"].strip()
            self.token_usage += res1.get("prompt_eval_count", 0) + res1.get("eval_count", 0)

            # Step 2: Extract the final answer
            if self.extract_final_answer:
                instruction = "Extract ONLY the final answer. Do not include any other text or explanation."
            else:
                instruction = "Synthesize the final response based on the reasoning chain. Provide the full complete text as requested by the original problem without repeating the reasoning steps."
                
            extract_prompt = f"Given the following reasoning:\n{reasoning}\n\n{instruction}"
            
            format_arg = None
            if schema is not None:
                try:
                    format_arg = schema.model_json_schema()
                except Exception:
                    pass

            if format_arg:
                res2 = ollama.generate(
                    model=self.model_name,
                    prompt=extract_prompt,
                    format=format_arg,
                    options={"temperature": 0, "seed": 42},
                )
                self.token_usage += res2.get("prompt_eval_count", 0) + res2.get("eval_count", 0)
                return schema.model_validate_json(res2["response"])
            else:
                res2 = ollama.generate(
                    model=self.model_name,
                    prompt=extract_prompt,
                    options={"temperature": 0, "seed": 42},
                )
                self.token_usage += res2.get("prompt_eval_count", 0) + res2.get("eval_count", 0)
                return res2["response"].strip()

        except Exception as e:
            print(f"Generation failed: {e}")
            return ""

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return self.model_name