import re
import json
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from deepeval.models.base_model import DeepEvalBaseLLM

class _ThoughtNode:
    def __init__(self, thought: str, parent=None, depth: int = 0):
        self.thought:     str              = thought
        self.parent:      "_ThoughtNode"   = parent
        self.depth:       int              = depth
        self.llm_score:   float            = 0.0
        self.sem_score:   float            = 0.0
        self.is_terminal: bool             = False
        self.children:    list             = []

    def full_state(self) -> list[str]:
        chain = []
        node  = self
        while node is not None:
            chain.append(node.thought)
            node = node.parent
        chain.reverse()
        return [t for t in chain if t != "[root]"]

class OtotussModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b-instruct",
        max_depth: int = 3,
        beam_width: int = 2,
        semantic_threshold: float = 0.50,
        n_thoughts: int = 4,
        min_keep: int = 1,
        early_termination_threshold: float = 9.3,
        embedding_model: str = "all-MiniLM-L6-v2",
        extract_final_answer: bool = True
    ):
        self.model_name = model_name
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.semantic_threshold = semantic_threshold
        self.n_thoughts = n_thoughts
        self.min_keep = min_keep
        self.early_termination_threshold = early_termination_threshold
        self.extract_final_answer = extract_final_answer
        self.token_usage = 0
        
        print(f"[OTOTUSS] Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)

    def load_model(self):
        return self.model_name

    def get_model_name(self) -> str:
        return f"OTOTUSS({self.model_name})"

    def generate(self, prompt: str, schema=None, **kwargs):
        # 1. Generate dynamic keywords
        dynamic_keywords = self._generate_dynamic_keywords(prompt)
        target_embeddings = self.embedding_model.encode(dynamic_keywords)

        # 2. Search
        best_node = self._search(prompt, target_embeddings)

        # 3. Synthesize
        return self._synthesize(prompt, best_node.full_state(), schema=schema)

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def _search(self, problem: str, target_embeddings: np.ndarray) -> _ThoughtNode:
        root = _ThoughtNode(thought="[root]", depth=0)
        active_set = [root]
        best_node = root
        best_score = -1.0

        for depth in range(1, self.max_depth + 1):
            all_thoughts_with_parent = []
            
            for node in active_set:
                thoughts = self._generate_thoughts(problem, node.full_state(), self.n_thoughts)
                for thought in thoughts:
                    all_thoughts_with_parent.append((thought, node))
            
            if not all_thoughts_with_parent:
                break

            # Semantic Gating
            scored_candidates = []
            for thought, parent in all_thoughts_with_parent:
                sem_score = self._compute_quality_similarity(thought, target_embeddings)
                scored_candidates.append((thought, parent, sem_score))
                
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            
            filtered_candidates = []
            for rank, (thought, parent, sem_score) in enumerate(scored_candidates):
                passes_threshold = sem_score >= self.semantic_threshold
                is_fallback = not passes_threshold and rank < self.min_keep
                if passes_threshold or is_fallback:
                    filtered_candidates.append((thought, parent, sem_score))

            # LLM Evaluation
            all_candidates = []
            for thought, parent, sem_score in filtered_candidates:
                child = _ThoughtNode(thought=thought, parent=parent, depth=depth)
                child.sem_score = sem_score
                child.llm_score = self._evaluate_thought_context(problem, child.full_state())
                
                parent.children.append(child)
                all_candidates.append(child)
                
                if child.llm_score > best_score:
                    best_score = child.llm_score
                    best_node = child
                    
            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x.llm_score, reverse=True)
            active_set = all_candidates[:self.beam_width]

            if active_set and active_set[0].llm_score >= self.early_termination_threshold:
                best_node = active_set[0]
                break
                
        return best_node

    def _generate_dynamic_keywords(self, problem: str) -> list[str]:
        prompt = f"""
Analyze the following problem statement and generate a list of 10-15 relevant, high-quality search keywords or short phrases. 
These keywords will be used to guide a semantic search, filtering for thoughts that are highly viable, efficient, and domain-appropriate.
Include both:
1. General qualitative solver attributes (e.g., efficient, optimized, feasible, sustainable).
2. Domain-specific, highly relevant terms for this particular problem.

Problem:
{problem}

Return ONLY the list of keywords/phrases separated by commas. No explanation, no numbering, no markdown, no preamble.
Example: efficient, optimized, sustainable, public transit, congestion pricing, urban planning
"""
        response = self._call_ollama(prompt, temperature=0.7)
        keywords = []
        for kw in response.split(","):
            clean_kw = kw.strip().replace('"', '').replace("'", "").lower()
            if clean_kw:
                keywords.append(clean_kw)
        return keywords

    def _compute_quality_similarity(self, thought: str, target_embeddings: np.ndarray, top_n: int = 5) -> float:
        from sklearn.metrics.pairwise import cosine_similarity
        thought_embedding = self.embedding_model.encode([thought])
        similarities = cosine_similarity(thought_embedding, target_embeddings)[0]
        return float(np.max(similarities))

    def _generate_thoughts(self, problem: str, state: list[str], k: int) -> list[str]:
        state_text = "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(state)) if state else "  (no reasoning steps yet)"
        
        prompt = f"""You are solving a problem step by step.

PROBLEM:
{problem}

CURRENT REASONING STATE:
{state_text}

Your task: propose exactly {k} distinct candidate next reasoning steps.

Requirements:
- Each step must be meaningfully different from the others
- Each step must directly advance toward solving the problem
- Each step should be 1-3 sentences
- Do NOT repeat any step already in the current reasoning state above

Return ONLY a valid JSON array of exactly {k} strings.
No markdown, no code fences, no explanation outside the array.
Example format: ["step one", "step two", "step three"]"""

        raw = self._call_ollama(prompt, temperature=0.7)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                thoughts = json.loads(match.group())
                return [str(t).strip() for t in thoughts if str(t).strip()]
            except json.JSONDecodeError:
                pass
        
        lines = [ln.strip(" -•123456789.)") for ln in raw.split("\n") if ln.strip()]
        return [ln for ln in lines if ln][:k]

    def _evaluate_thought_context(self, problem: str, state: list[str]) -> float:
        state_text = "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(state))
        
        prompt = f"""
Problem:
{problem}

Reasoning so far:
{state_text}

Evaluate the quality and correctness of the reasoning above for solving the problem.
Consider: logical progression, mathematical accuracy, relevance, and completeness of steps.
Respond with ONLY the numerical rating (between 1.0 and 10.0, e.g. 7.5). Do not provide any justification, explanation, or markdown.
"""
        raw = self._call_ollama(prompt, temperature=0.1)
        
        all_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw)
        if all_numbers:
            try:
                return max(1.0, min(10.0, float(all_numbers[-1])))
            except ValueError:
                pass
        
        return 1.0

    def _synthesize(self, problem: str, best_path: list[str], schema=None):
        path_text = "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(best_path)) if best_path else "(no reasoning steps yet)"

        if self.extract_final_answer:
            instruction = "Extract and output ONLY the final answer. Do not include any other text or explanation."
        else:
            instruction = "Synthesize a highly detailed and comprehensive final response based on the reasoning chain. Provide a thorough explanation covering all aspects of the solution without repeating the step-by-step reasoning."

        prompt = f"""You have reasoned through the following problem step by step.
Now write a clear, complete, and correct final answer.

PROBLEM:
{problem}

REASONING CHAIN (best path found by Tree of Thoughts search):
{path_text}

{instruction}"""

        format_arg = None
        if schema is not None:
            try:
                format_arg = schema.model_json_schema()
            except Exception:
                pass

        raw_response = self._call_ollama(prompt, temperature=0.3, format_arg=format_arg)
        
        if schema is not None and format_arg is not None:
            try:
                return schema.model_validate_json(raw_response)
            except Exception as e:
                print(f"[OTOTUSS] Schema validation failed: {e}")
                return raw_response
        return raw_response

    def _call_ollama(self, prompt: str, temperature: float = 0.7, format_arg=None) -> str:
        try:
            kwargs = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "seed":        42,
                }
            }
            if format_arg:
                kwargs["format"] = format_arg

            response = ollama.generate(**kwargs)

            prompt_tokens     = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            self.token_usage += prompt_tokens + completion_tokens

            return response["response"].strip()
        except Exception as e:
            print(f"[OTOTUSS] Ollama call failed: {e}")
            return ""
