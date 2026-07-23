import ollama
from deepeval.models.base_model import DeepEvalBaseLLM


class CoTModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.token_usage = 0

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        cot_prompt = f"{prompt} \nWork through the problem step by step before giving the final answer."

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=cot_prompt,
                options={
                    "temperature": 0,
                    "seed": 42,
                    "think": False,
                },
            )

            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)

            self.token_usage += prompt_tokens + completion_tokens

            return response["response"].strip()

        except Exception as e:
            print(f"Generation failed: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name