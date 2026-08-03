import ollama
from deepeval.models.base_model import DeepEvalBaseLLM

class BaseModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str, extract_final_answer: bool = True):
        self.model_name = model_name
        self.extract_final_answer = extract_final_answer
        self.token_usage = 0

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema=None, **kwargs):
        try:
            format_arg = None
            if schema is not None:
                try:
                    format_arg = schema.model_json_schema()
                except Exception:
                    pass

            if format_arg:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    format=format_arg,
                    options={
                        "temperature": 0,
                        "seed": 42,
                    },
                )
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                self.token_usage += prompt_tokens + completion_tokens
                return schema.model_validate_json(response["response"])
            else:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0,
                        "seed": 42,
                    },
                )
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                self.token_usage += prompt_tokens + completion_tokens
                return response["response"].strip()

        except Exception as e:
            print(f"Generation failed: {e}")
            return ""

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return self.model_name