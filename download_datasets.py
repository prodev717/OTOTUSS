from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import GSM8K, BoolQ

class DummyModel(DeepEvalBaseLLM):
    def load_model(self):
        return None

    def generate(self, prompt):
        return "A"

    async def a_generate(self, prompt):
        return self.generate(prompt)

    def get_model_name(self):
        return "dummy"

model = DummyModel()

benchmarks = [
    GSM8K(),
    BoolQ(),
]

for benchmark in benchmarks:
    try:
        benchmark.evaluate(model=model)
    except Exception:
        # Ignore evaluation errors—the dataset is already downloaded.
        pass

print("Datasets downloaded.")