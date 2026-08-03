import re
import json
import math
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from deepeval.models.base_model import DeepEvalBaseLLM


class _SSDPNode:


    def __init__(self, thought: str, parent=None, depth: int = 0):
        self.thought:      str                   = thought
        self.parent:       "_SSDPNode"           = parent
        self.depth:        int                   = depth
        self.children:     list                  = []

        # reward model score ϕ(c) — set after expansion + scoring
        self.phi:          float                 = 0.0

        # MCTS statistics for UCB  (paper Section 3.1)
        self.visit_count:  int                   = 0
        self.cumul_reward: float                 = 0.0

        # embedding vector s(c) = E(text(c))  (paper Section 3.2 step 1)
        # set during the EMBED stage of the loop
        self.embedding:    np.ndarray            = None

        # how many siblings were merged into this node  (diagnostic)
        self.merged_count: int                   = 0

        # is this a terminal (complete solution) node?
        self.is_terminal:  bool                  = False

    def full_state(self) -> list[str]:
    
        chain, node = [], self
        while node is not None:
            chain.append(node.thought)
            node = node.parent
        chain.reverse()
        return [t for t in chain if t != "[root]"]

    def ucb(self, parent_visits: int, w: float) -> float:

        if self.visit_count == 0:
            return float("inf")
        exploit = self.cumul_reward / self.visit_count
        explore = w * math.sqrt(1.0 + parent_visits) / (1.0 + self.visit_count)
        return exploit + explore * self.phi

    def __repr__(self):
        preview = self.thought[:50].replace("\n", " ")
        return (
            f"_SSDPNode(depth={self.depth}, phi={self.phi:.2f}, "
            f"merged={self.merged_count}, '{preview}')"
        )


class SSDPModel(DeepEvalBaseLLM):


    def __init__(
        self,
        model_name:            str   = "qwen2.5-coder:7b-instruct",
        similarity_threshold:  float = 0.75,        # τ   — paper default
        beam_size:             int   = 4,            # b   — paper default
        max_depth:             int   = 3,            # T
        beam_width:            int   = 3,            # frontier width
        exploration_weight:    float = 1/math.sqrt(2),  # w — paper default
        early_stopping_factor: float = 0.8,          # λes — paper default
        embedding_model:       str   = "all-MiniLM-L6-v2",  # Appendix A.6
        max_rollouts:          int   = 20,           # Rmax — paper default
        extract_final_answer:  bool  = True,
        seed: int = 42
    ):
        self.model_name            = model_name
        self.tau                   = similarity_threshold
        self.beam_size             = beam_size
        self.max_depth             = max_depth
        self.beam_width            = beam_width
        self.w                     = exploration_weight
        self.lambda_es             = early_stopping_factor
        self.max_rollouts          = max_rollouts
        self.extract_final_answer  = extract_final_answer
        self.seed = seed

       
        self.token_usage: int = 0

       
        self._nodes_generated: int = 0
        self._nodes_pruned:    int = 0

        print(f"  [SSDP] Loading embedding model: {embedding_model}...", end=" ", flush=True)
        self._embedder = SentenceTransformer(embedding_model)
        print("ready.")

  

    def load_model(self):
    
        return self.model_name

    def get_model_name(self) -> str:
       
        return f"SSDP(τ={self.tau}, {self.model_name})"

    

    def generate(self, prompt: str, schema=None, **kwargs):

        # reset per-call diagnostics
        self._nodes_generated = 0
        self._nodes_pruned    = 0

        # run the SSDP search loop (Appendix A.3)
        best_node = self._ssdp_search(prompt)

        # synthesize a clean final answer from the best path found
        return self._synthesize(prompt, best_node.full_state(), schema=schema)

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        """Async wrapper required by DeepEvalBaseLLM."""
        return self.generate(prompt, schema=schema, **kwargs)

 
    def _ssdp_search(self, problem: str) -> "_SSDPNode":
  
        # initialise root
        root            = _SSDPNode(thought="[root]", depth=0)
        root.phi        = 0.5
        root.visit_count = 1

        # Nall — all nodes ever added to the tree
        all_nodes: list[_SSDPNode] = [root]

        # frontier — leaf nodes eligible for expansion
        frontier: list[_SSDPNode] = [root]

        # track best terminal found
        best_terminal: _SSDPNode  = None

        rollout = 0

        for depth in range(1, self.max_depth + 1):

            if not frontier or rollout >= self.max_rollouts:
                break

            # ── STEP 1: SELECT  (paper Section 3.1 step 1) ────────────
            # "select up to k promising leaf nodes using UCB policy"
            parent_visits  = sum(n.visit_count for n in all_nodes)
            selected       = self._select(frontier, self.beam_width, parent_visits)
            next_frontier: list[_SSDPNode] = []

            for node in selected:
                rollout += 1
                if rollout > self.max_rollouts:
                    break

                # ── STEP 2: EXPAND  (paper Section 3.1 step 2) ────────
                # "generate b candidate children via temperature sampling"
                state    = node.full_state()
                raw_thoughts = self._generate_thoughts(problem, state, self.beam_size)
                if not raw_thoughts:
                    continue

                self._nodes_generated += len(raw_thoughts)

                # ── STEP 2 cont: SCORE + EMBED each child ─────────────
                # "ϕ(c) ← PRM(c)"      score with reward model
                # "s(c) ← E(decode(xc))" compute embedding
                children: list[_SSDPNode] = []
                texts_for_embed           = []

                for text in raw_thoughts:
                    child             = _SSDPNode(thought=text, parent=node, depth=depth)
                    child.phi         = self._score(problem, child.full_state())
                    child.is_terminal = self._check_terminal(problem, child.full_state())
                    children.append(child)
                    texts_for_embed.append(text)

                # batch-embed all children at once (efficient)
                embeddings = self._embed_batch(texts_for_embed)
                for child, emb in zip(children, embeddings):
                    child.embedding = emb

                # early stopping (Appendix A.4):
                # "if ϕ(n) < θes = λes · ϕ̄_explore, terminate rollout"
                explored_phis = [n.phi for n in all_nodes if n.phi > 0]
                phi_mean      = sum(explored_phis) / max(len(explored_phis), 1)
                theta_es      = self.lambda_es * phi_mean

                children = [c for c in children if c.phi >= theta_es]

                # ── STEP 3: CLUSTERANDPRUNE(Cn, τ) ────────────────────
                # This is the core SSDP contribution (paper Section 3.2)
                survivors = self._cluster_and_prune(children)

                pruned_count = len(children) - len(survivors)
                self._nodes_pruned += pruned_count

                # attach survivors to the tree
                node.children = survivors
                all_nodes.extend(survivors)

                # ── check for terminals ────────────────────────────────
                for child in survivors:
                    if child.is_terminal:
                        best_terminal = child
                        break

                if best_terminal:
                    break

                next_frontier.extend(survivors)

            if best_terminal:
                break

            # ── STEP 4: BACKPROPAGATE  (paper Section 3.1 step 4) ─────
            # "propagate reward up the tree to update Qu, Nu of ancestors"
            for node in selected:
                if node.children:
                    best_phi = max(c.phi for c in node.children)
                    self._backprop(node, best_phi)

            # update frontier — sort by phi, keep top beam_width
            next_frontier.sort(key=lambda n: n.phi, reverse=True)
            frontier = next_frontier[:self.beam_width]

        # return best terminal if found, else best node by phi
        if best_terminal:
            return best_terminal

        real_nodes = [n for n in all_nodes if n.thought != "[root]"]
        if not real_nodes:
            return root
        return max(real_nodes, key=lambda n: n.phi)

 

    def _select(
        self,
        frontier:      list["_SSDPNode"],
        k:             int,
        parent_visits: int,
    ) -> list["_SSDPNode"]:
     
        scored = sorted(
            frontier,
            key=lambda n: n.ucb(parent_visits, self.w),
            reverse=True,
        )
        return scored[:k]

   

    def _generate_thoughts(
        self,
        problem: str,
        state:   list[str],
        b:       int,
    ) -> list[str]:

        state_text = (
            "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(state))
            if state else "  (no steps yet — this is the first move)"
        )

        prompt = f"""You are solving a problem step by step.

PROBLEM:
{problem}

REASONING SO FAR:
{state_text}

Propose exactly {b} DISTINCT candidate next reasoning steps.

Rules:
- Every step must be meaningfully different from the others
- Every step must advance toward the solution
- Each step should be 1-3 sentences
- Do NOT repeat steps already shown above

Respond with ONLY a valid JSON array of {b} strings.
No markdown fences, no explanation outside the array.
Example: ["step one", "step two", "step three", "step four"]"""

        raw = self._call_ollama(prompt, temperature=0.8)

        # strip markdown fences qwen sometimes adds
        raw   = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                thoughts = json.loads(match.group())
                return [str(t).strip() for t in thoughts if str(t).strip()]
            except json.JSONDecodeError:
                pass

        # fallback: split by newlines
        lines = [ln.strip(" -•0123456789.)") for ln in raw.split("\n") if ln.strip()]
        return [ln for ln in lines if ln][:b]


    def _score(self, problem: str, state: list[str]) -> float:
 
        state_text = "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(state))

        prompt = f"""You are evaluating how promising a reasoning path is.

PROBLEM:
{problem}

REASONING PATH:
{state_text}

Rate this reasoning path from 0 to 10:
  0  = completely wrong or impossible to reach the solution
  3  = poor, major errors or off-track
  5  = reasonable but uncertain
  8  = strong, likely correct approach
  10 = excellent, almost certainly correct

Respond with ONLY a single integer (0-10). Nothing else."""

        raw   = self._call_ollama(prompt, temperature=0.1)
        match = re.search(r'\b(\d{1,2})\b', raw.strip())
        if match:
            return min(max(int(match.group(1)), 0), 10) / 10.0
        return 0.5   # neutral fallback

   
    def _cluster_and_prune(
        self,
        children: list["_SSDPNode"],
    ) -> list["_SSDPNode"]:
    
        if not children:
            return []

        # single child → no merging needed
        if len(children) == 1:
            return children

       
        clusters: list[list["_SSDPNode"]] = []

        for node in children:
            if node.embedding is None:
                # no embedding (edge case) — start its own cluster
                clusters.append([node])
                continue

            placed = False
            for cluster in clusters:
                seed = cluster[0]
                if seed.embedding is None:
                    continue
                # cosine similarity — both embeddings are ℓ2-normalised
                # so dot product = cosine similarity  [paper Section 3.2 step 1]
                sim = float(np.dot(node.embedding, seed.embedding))
                if sim >= self.tau:
                    cluster.append(node)
                    placed = True
                    break

            if not placed:
                clusters.append([node])

        
        representatives: list["_SSDPNode"] = []

        for cluster in clusters:
            # pick the node with the highest reward score
            best = max(cluster, key=lambda n: n.phi)

            # record how many siblings were merged into this representative
            # (diagnostic — helps track SSDP efficiency)
            best.merged_count = len(cluster) - 1

            representatives.append(best)

        return representatives


    def _backprop(self, node: "_SSDPNode", reward: float):

        current = node
        while current is not None:
            current.visit_count  += 1
            current.cumul_reward += reward
            current = current.parent



    def _check_terminal(self, problem: str, state: list[str]) -> bool:
     
        if not state:
            return False

        state_text = "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(state))

        prompt = f"""Has the following reasoning path FULLY and CORRECTLY solved the problem?

PROBLEM:
{problem}

REASONING PATH:
{state_text}

Answer YES only if:
  - The problem is completely solved
  - A clear, correct final answer is stated
  - No important steps are missing

Answer NO if the reasoning is still in progress.

Respond with ONLY: YES or NO"""

        raw = self._call_ollama(prompt, temperature=0.0)
        return "YES" in raw.upper()

 

    def _synthesize(self, problem: str, best_path: list[str], schema=None):

        if not best_path:
            path_text = "(no reasoning steps)"
        else:
            path_text = "\n".join(
                f"  Step {i+1}: {t}" for i, t in enumerate(best_path)
            )

        if self.extract_final_answer:
            instruction = "Extract and output ONLY the final answer. Do not include any other text or explanation."
        else:
            instruction = "Synthesize a highly detailed and comprehensive final response based on the reasoning chain. Provide a thorough explanation covering all aspects of the solution without repeating the step-by-step reasoning."

        prompt = f"""You have reasoned through a problem using Tree of Thoughts search
with Semantic Similarity Based Dynamic Pruning (SSDP).
Now write a clear, complete, and correct final answer.

PROBLEM:
{problem}

BEST REASONING CHAIN FOUND:
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
                print(f"[SSDP] Schema validation failed: {e}")
                return raw_response
        return raw_response


    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Batch embed a list of texts using the frozen sentence-transformer.

        Paper Section 3.2 step 1:
            s(c) = E(text(c)), ℓ2-normalised

        Args:
            texts : list of strings to embed

        Returns:
            list of ℓ2-normalised numpy arrays (one per text)
        """
        if not texts:
            return []

        # encode all at once — more efficient than one-by-one
        vectors = self._embedder.encode(texts, convert_to_numpy=True)

        # ℓ2-normalise each vector  [paper Section 3.2 step 1]
        # "normalised to facilitate efficient cosine similarity calculations"
        normalised = []
        for vec in vectors:
            norm = np.linalg.norm(vec)
            normalised.append(vec / norm if norm > 0 else vec)
        return normalised

    def _embed_one(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns ℓ2-normalised vector."""
        result = self._embed_batch([text])
        return result[0] if result else np.zeros(384)

  

    def _call_ollama(self, prompt: str, temperature: float = 0.7, format_arg=None) -> str:
 
        try:
            kwargs_dict = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "seed": self.seed,
                }
            }
            if format_arg:
                kwargs_dict["format"] = format_arg

            response = ollama.generate(**kwargs_dict)

            # token tracking — identical to base.py and cot.py
            prompt_tokens     = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            self.token_usage += prompt_tokens + completion_tokens

            return response["response"].strip()

        except Exception as e:
            print(f"[SSDP] Ollama call failed: {e}")
            return ""



    def ssdp_stats(self) -> dict:
      
        explored = self._nodes_generated - self._nodes_pruned
        reduction = (
            round(self._nodes_pruned / self._nodes_generated * 100, 1)
            if self._nodes_generated > 0 else 0.0
        )
        return {
            "nodes_generated": self._nodes_generated,
            "nodes_pruned":    self._nodes_pruned,
            "nodes_explored":  explored,
            "reduction_pct":   reduction,
            "token_usage":     self.token_usage,
        }