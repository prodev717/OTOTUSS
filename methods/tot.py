import re
import json
import ollama
from deepeval.models.base_model import DeepEvalBaseLLM


class _ThoughtNode:
    def __init__(self, thought: str, parent=None, depth: int = 0):
        self.thought:     str              = thought
        self.parent:      "_ThoughtNode"   = parent
        self.depth:       int              = depth
        self.value:       float            = 0.0
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

    def __repr__(self):
        preview = self.thought[:50].replace("\n", " ")
        return f"_ThoughtNode(depth={self.depth}, value={self.value:.2f}, '{preview}')"
 

class ToTModel(DeepEvalBaseLLM):


    def __init__(
        self,
        model_name:       str   = "qwen2.5-coder:7b-instruct",
        search_strategy:  str   = "bfs",
        n_thoughts:       int   = 5,
        max_depth:        int   = 3,
        beam_width:       int   = 5,
        value_threshold:  float = 0.4,
        n_evals:          int   = 3,
        extract_final_answer: bool = True,
        seed: int = 42
    ):
        self.model_name      = model_name
        self.search_strategy = search_strategy   # "bfs" or "dfs"
        self.n_thoughts      = n_thoughts        # k in the paper
        self.max_depth       = max_depth         # T in the paper
        self.beam_width      = beam_width        # b in the paper
        self.value_threshold = value_threshold   # vth in the paper
        self.n_evals         = n_evals           # number of value samples to average
        self.extract_final_answer = extract_final_answer
        self.seed = seed

        # token usage — accumulates across the full generate() call
        # (matches base.py / cot.py pattern exactly)
        self.token_usage: int = 0

    # ── DeepEval required methods ────────────────────────────────────

    def load_model(self):
        """Returns the model identifier (required by DeepEvalBaseLLM)."""
        return self.model_name

    def get_model_name(self) -> str:
        """Returns a human-readable model name."""
        return f"ToT-{self.search_strategy.upper()}({self.model_name})"

    # ── Main public entry point ───────────────────────────────────────

    def generate(self, prompt: str, schema=None, **kwargs):
 
        if self.search_strategy == "bfs":
            best_node = self._bfs(prompt)
        elif self.search_strategy == "dfs":
            best_node = self._dfs_entry(prompt)
        else:
            raise ValueError(
                f"Unknown search_strategy '{self.search_strategy}'. "
                "Use 'bfs' or 'dfs'."
            )

        # build the best reasoning path and synthesize a final answer
        best_path = best_node.full_state()
        return self._synthesize(prompt, best_path, schema=schema)

    async def a_generate(self, prompt: str, schema=None, **kwargs):
     
        return self.generate(prompt, schema=schema, **kwargs)

    # ═════════════════════════════════════════════════════════════════
    # ALGORITHM 1 — BFS  (paper Section 3 / Algorithm 1)
    # ═════════════════════════════════════════════════════════════════
    #
    # Pseudocode from the paper:
    #   S0 ← {x}
    #   for t = 1 … T:
    #       S't ← {[s, z] | s ∈ St-1, z ∈ G(pθ, s, k)}
    #       Vt  ← V(pθ, S't)
    #       St  ← argmax_{S ⊂ S't, |S|=b}  Σ_{s∈S} Vt(s)
    #   return G(pθ, argmax_{s∈ST} VT(s), 1)
    #
    # Translation to our code:
    #   S0       = [root node]
    #   G(...)   = _generate_thoughts()   → produces k candidate children
    #   V(...)   = _evaluate_thought()    → scores each child 0.0–1.0
    #   argmax b = sort by value, keep top beam_width
    #   return   = _synthesize() on the best leaf

    def _bfs(self, problem: str) -> "_ThoughtNode":
    
        # S0 — initial frontier with just the root
        root     = _ThoughtNode(thought="[root]", depth=0)
        frontier = [root]   # St-1 in the paper

        for depth in range(1, self.max_depth + 1):

            # S't — expand every node in the frontier
            # "S't ← {[s, z] | s ∈ St-1, z ∈ G(pθ, s, k)}"
            all_children: list[_ThoughtNode] = []

            for node in frontier:
                state = node.full_state()   # s = [x, z1, ..., z_{t-1}]

                # G(pθ, s, k) — generate k candidate next thoughts
                thoughts = self._generate_thoughts(problem, state, self.n_thoughts)

                for thought_text in thoughts:
                    child = _ThoughtNode(
                        thought=thought_text,
                        parent=node,
                        depth=depth,
                    )

                    # V(pθ, S)(s) — evaluate this child
                    # Paper samples value n_evals times and averages
                    child.value       = self._evaluate_thought(problem, child.full_state())
                    child.is_terminal = self._check_terminal(problem, child.full_state())

                    node.children.append(child)
                    all_children.append(child)

                    # early exit if a complete solution was found
                    if child.is_terminal:
                        return child

            if not all_children:
                break

            # St ← argmax_{S ⊂ S't, |S|=b} Σ Vt(s)
            # Keep only the top-b children by value score
            all_children.sort(key=lambda n: n.value, reverse=True)
            frontier = all_children[:self.beam_width]

        # "return G(pθ, argmax_{s∈ST} VT(s), 1)"
        # Return the best node from the final frontier
        if not frontier:
            return root
        return max(frontier, key=lambda n: n.value)

    # ═════════════════════════════════════════════════════════════════
    # ALGORITHM 2 — DFS  (paper Section 3 / Algorithm 2)
    # ═════════════════════════════════════════════════════════════════
    #
    # Pseudocode from the paper:
    #   DFS(s, t):
    #     if t > T: record output G(pθ, s, 1)
    #     for s' in G(pθ, s, k):   ← sorted candidates
    #       if V(pθ, {s'})(s) > vth:   ← pruning
    #         DFS(s', t+1)
    #
    # Key paper details:
    #   - Explore candidates in sorted (best-first) order
    #   - Prune branches where value ≤ vth (threshold)
    #   - Backtrack to parent when a branch is pruned or exhausted

    def _dfs_entry(self, problem: str) -> "_ThoughtNode":
        """Entry point for DFS — sets up root and calls recursive DFS."""
        root          = _ThoughtNode(thought="[root]", depth=0)
        self._best_dfs: _ThoughtNode = root   # tracks best found node

        self._dfs(problem, root, depth=1)
        return self._best_dfs

    def _dfs(self, problem: str, node: "_ThoughtNode", depth: int) -> bool:
       
        if depth > self.max_depth:
            if node.value > self._best_dfs.value:
                self._best_dfs = node
            return False

        state = node.full_state()   # s = [x, z1, ..., z_{t-1}]

        # G(pθ, s, k) — generate k candidates and score them
        thoughts = self._generate_thoughts(problem, state, self.n_thoughts)

        # build candidate children with their values
        candidates: list[_ThoughtNode] = []
        for thought_text in thoughts:
            child             = _ThoughtNode(thought=thought_text, parent=node, depth=depth)
            child.value       = self._evaluate_thought(problem, child.full_state())
            child.is_terminal = self._check_terminal(problem, child.full_state())
            node.children.append(child)
            candidates.append(child)

        # "for s' in G(pθ, s, k): ← sorted candidates"
        # Paper explores best candidates first
        candidates.sort(key=lambda c: c.value, reverse=True)

        for candidate in candidates:
            # check terminal BEFORE pruning — a complete answer should always be returned
            if candidate.is_terminal:
                self._best_dfs = candidate
                return True   # propagate up to stop the whole search

            # "if V(pθ, {s'})(s) > vth: ← pruning"
            # Prune branches whose value is at or below the threshold (backtrack)
            if candidate.value <= self.value_threshold:
                continue   # backtrack — try next candidate

            # track best seen so far (in case no terminal is ever found)
            if candidate.value > self._best_dfs.value:
                self._best_dfs = candidate

            # recurse into this candidate (DFS(s', t+1))
            found = self._dfs(problem, candidate, depth + 1)
            if found:
                return True   # terminal was found deeper — propagate up

        return False   # exhausted all candidates at this level → backtrack

    # ═════════════════════════════════════════════════════════════════
    # THOUGHT GENERATOR  G(pθ, s, k)
    # Paper Section 3, Point 2b — "propose prompt" strategy
    # ═════════════════════════════════════════════════════════════════
    #
    # Paper quote:
    #   "Propose thoughts sequentially using a 'propose prompt':
    #    [z(1), ..., z(k)] ~ p_propose(z(1...k)_{i+1} | s)
    #    This works better when the thought space is more constrained
    #    (e.g. each thought is a line), so proposing different thoughts
    #    in the same context avoids duplication."
    #
    # We use a single call that asks the model to propose k distinct
    # next steps and return them as a JSON array — matching the paper's
    # "propose prompt" where all k thoughts are generated in one context.

    def _generate_thoughts(
        self,
        problem: str,
        current_state: list[str],
        k: int,
    ) -> list[str]:

        # build the state description for the prompt
        if current_state:
            state_text = "\n".join(
                f"  Step {i+1}: {t}" for i, t in enumerate(current_state)
            )
        else:
            state_text = "  (no reasoning steps yet)"

        # ── Propose Prompt ────────────────────────────────────────────
        # Mirrors Figure 2(a) from the paper: given the current partial
        # solution state s, list k distinct possible next steps.
        propose_prompt = f"""You are solving a problem step by step.

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

        raw = self._call_ollama(propose_prompt, temperature=0.7)

        # parse the JSON array robustly
        # (the model sometimes wraps output in ```json ... ```)
        raw   = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                thoughts = json.loads(match.group())
                return [str(t).strip() for t in thoughts if str(t).strip()]
            except json.JSONDecodeError:
                pass

        # fallback: split by newlines if JSON failed
        lines = [ln.strip(" -•123456789.)") for ln in raw.split("\n") if ln.strip()]
        return [ln for ln in lines if ln][:k]

    # ═════════════════════════════════════════════════════════════════
    # STATE EVALUATOR  V(pθ, S)  — "value each state independently"
    # Paper Section 3, Point 3a
    # ═════════════════════════════════════════════════════════════════
    #
    # Paper quote:
    #   "Value each state independently: V(pθ, S)(s) ~ p_value(v|s)
    #    where a value prompt reasons about the state s to generate
    #    a scalar value v (e.g. 1-10) or a classification
    #    (e.g. sure/likely/impossible)."
    #
    # Paper Section 4.1 (Game of 24):
    #   "We prompt LM to evaluate each thought candidate as
    #    'sure/maybe/impossible' with regard to reaching 24."
    #   "We sample values 3 times for each thought."  ← n_evals=3
    #
    # We implement a 0-10 scalar value prompt (the paper's first option)
    # and average across n_evals samples (paper samples 3 times).

    def _evaluate_thought(self, problem: str, state: list[str]) -> float:

        state_text = "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(state))

        # ── Value Prompt ──────────────────────────────────────────────
        # Mirrors Figure 2(b) from the paper: evaluate how promising
        # this partial reasoning state is for reaching the solution.
        value_prompt = f"""You are evaluating how promising a reasoning path is for solving a problem.

PROBLEM:
{problem}

REASONING PATH SO FAR:
{state_text}

Rate this reasoning path on a scale from 0 to 10:
  0  = impossible to reach the correct solution from here
  3  = unlikely, significant errors or missing steps
  5  = possible but uncertain, partial progress
  8  = likely correct, good progress toward solution
  10 = certain to reach or already at the correct solution

Respond with ONLY a single integer from 0 to 10. Nothing else."""

        # sample n_evals times and average (paper samples 3 times)
        scores = []
        for _ in range(self.n_evals):
            raw   = self._call_ollama(value_prompt, temperature=0.1)
            match = re.search(r'\b(\d{1,2})\b', raw.strip())
            if match:
                score = int(match.group(1))
                scores.append(min(max(score, 0), 10))

        if not scores:
            return 0.5   # neutral fallback if all parses failed

        # normalise from [0, 10] to [0.0, 1.0]
        return sum(scores) / len(scores) / 10.0

    # ═════════════════════════════════════════════════════════════════
    # TERMINAL CHECK
    # Paper: "if t > T: record output G(pθ, s, 1)"
    # We also check mid-search if a complete solution has been reached.
    # ═════════════════════════════════════════════════════════════════

    def _check_terminal(self, problem: str, state: list[str]) -> bool:
        """
        Ask the model whether the current reasoning state constitutes
        a complete, correct solution to the problem.

        Paper reference: the paper uses the value evaluator to detect
        "sure" states (complete solutions). We use an explicit YES/NO
        check for clarity.

        Returns:
            True if the model judges the problem fully solved.
        """
        state_text = "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(state))

        terminal_prompt = f"""Determine if the following reasoning path has FULLY and CORRECTLY solved the problem.

PROBLEM:
{problem}

REASONING PATH:
{state_text}

Answer YES only if:
  - The problem is completely solved
  - A clear, correct final answer has been stated
  - No important reasoning steps are missing

Answer NO if the reasoning is still in progress or incomplete.

Respond with ONLY: YES or NO"""

        raw = self._call_ollama(terminal_prompt, temperature=0.0)
        return "YES" in raw.upper()

    # ═════════════════════════════════════════════════════════════════
    # FINAL ANSWER SYNTHESIZER
    # Paper: "return G(pθ, argmax_{s∈ST} VT(s), 1)"
    # The paper calls generate one final time on the best state to
    # produce the output. We do the same.
    # ═════════════════════════════════════════════════════════════════

    def _synthesize(self, problem: str, best_path: list[str], schema=None):
      
        if not best_path:
            path_text = "(no reasoning steps yet)"
        else:
            path_text = "\n".join(
                f"  Step {i+1}: {t}" for i, t in enumerate(best_path)
            )

        if self.extract_final_answer:
            instruction = "Extract and output ONLY the final answer. Do not include any other text or explanation."
        else:
            instruction = "Synthesize a highly detailed and comprehensive final response based on the reasoning chain. Provide a thorough explanation covering all aspects of the solution without repeating the step-by-step reasoning."

        synthesis_prompt = f"""You have reasoned through the following problem step by step.
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

        raw_response = self._call_ollama(synthesis_prompt, temperature=0.3, format_arg=format_arg)
        
        if schema is not None and format_arg is not None:
            try:
                return schema.model_validate_json(raw_response)
            except Exception as e:
                print(f"[ToT] Schema validation failed: {e}")
                return raw_response
        return raw_response

    # ═════════════════════════════════════════════════════════════════
    # PRIVATE: single Ollama call + token tracking
    # Matches the exact pattern used in base.py and cot.py
    # ═════════════════════════════════════════════════════════════════

    def _call_ollama(self, prompt: str, temperature: float = 0.7, format_arg=None) -> str:
 
        try:
            kwargs = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "seed": self.seed,
                }
            }
            if format_arg:
                kwargs["format"] = format_arg

            response = ollama.generate(**kwargs)

            # token tracking — same as base.py and cot.py
            prompt_tokens     = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            self.token_usage += prompt_tokens + completion_tokens

            return response["response"].strip()

        except Exception as e:
            print(f"[ToT] Ollama call failed: {e}")
            return ""