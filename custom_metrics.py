from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from methods.base import BaseModel

judge_model_name = "qwen3:8b" 
judge_model = BaseModel(judge_model_name)

# -----------------------------
# Metrics
# -----------------------------

relevancy_metric = AnswerRelevancyMetric(
    threshold=0.5,
    model=judge_model
)

creativity_metric = GEval(
    name="Creativity",
    model=judge_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    criteria="""
Evaluate how creative and original the writing is.

Consider:
- originality
- imagination
- vivid imagery
- uniqueness of ideas
- engaging language

Return a score from 0 to 1.
"""
)

coherence_metric = GEval(
    name="Coherence",
    model=judge_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    criteria="""
Evaluate how coherent the writing is.

Consider:
- logical flow
- organization
- smooth transitions
- readability
- consistency

Return a score from 0 to 1.
"""
)

emotion_metric = GEval(
    name="Emotional Depth",
    model=judge_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    criteria="""
Evaluate the emotional impact of the writing.

Consider:
- emotional engagement
- depth of feeling
- expressiveness
- ability to evoke emotions

Return a score from 0 to 1.
"""
)

quality_metric = GEval(
    name="Writing Quality",
    model=judge_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    criteria="""
Evaluate the overall writing quality.

Consider:
- grammar
- vocabulary
- sentence structure
- clarity
- fluency
- overall polish

Return a score from 0 to 1.
"""
)