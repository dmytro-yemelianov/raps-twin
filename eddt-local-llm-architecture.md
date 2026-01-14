# Local LLM Architecture for EDDT Agent Decision-Making

## The Cost Problem

Running a simulation with 50 agents, 15-minute ticks, over 6 months of simulated time:

```
Ticks per day:     32 (8 working hours × 4 ticks/hour)
Working days:      130 (6 months)
Total ticks:       4,160 per agent
Agents:            50
Total LLM calls:   208,000

At Claude pricing (~$0.003/1K input + $0.015/1K output tokens):
  Avg prompt: 2,000 tokens input, 500 tokens output
  Cost per call: $0.006 + $0.0075 = $0.0135
  Total cost: 208,000 × $0.0135 = $2,808 per simulation run

That's ~$3K just to simulate one scenario once.
```

**Solution**: Hierarchical LLM architecture with local models handling 95%+ of decisions.

---

## Decision Hierarchy

Not all agent decisions require the same intelligence level:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DECISION COMPLEXITY PYRAMID                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                            ┌───────┐                                    │
│                           /  TIER  \         Cloud LLM (Claude/GPT)    │
│                          /    3     \        ~2% of decisions           │
│                         /  Complex   \       Novel situations           │
│                        / Reasoning    \      Multi-agent negotiation    │
│                       /________________\     Creative problem-solving   │
│                      /                  \                               │
│                     /       TIER 2       \   Local 7-13B Model         │
│                    /    Contextual        \  ~15% of decisions          │
│                   /     Decisions          \ Task prioritization        │
│                  /                          \Communication content      │
│                 /____________________________\Blocker resolution        │
│                /                              \                         │
│               /           TIER 1               \ Local 1-3B Model      │
│              /       Routine Decisions          \~83% of decisions     │
│             /                                    \Next action selection │
│            /    Rule-following, pattern-matching  \State transitions   │
│           /________________________________________\Tool selection     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Routine Decisions (Local 1-3B Model)

**What**: Predictable, pattern-based decisions that follow clear rules.

**Examples**:
- "I have a design task assigned → start working on it"
- "File upload complete → start translation"  
- "Review requested → check my queue and accept/defer"
- "Current task blocked → switch to next priority item"
- "End of work day → stop working"

**Why small model works**: These decisions can be reduced to classification or simple template completion. The context is structured and the output space is constrained.

### Tier 2: Contextual Decisions (Local 7-13B Model)

**What**: Decisions requiring understanding of context, priorities, and relationships.

**Examples**:
- "Multiple tasks available, which do I prioritize?"
- "How do I phrase this message to the reviewer?"
- "Should I escalate this blocker or wait longer?"
- "What approach should I take for this design task?"

**Why medium model works**: These require some reasoning but within well-defined problem spaces. Few-shot prompting with examples handles most cases.

### Tier 3: Complex Reasoning (Cloud LLM)

**What**: Novel situations, multi-agent coordination, creative problem-solving.

**Examples**:
- "Two agents have conflicting priorities, how to resolve?"
- "Unexpected failure occurred, what's the recovery plan?"
- "Client changed requirements mid-project, reassess approach"
- "Generate realistic excuse for why task is delayed" (for simulation realism)

**Why cloud model needed**: These require genuine reasoning, world knowledge, and nuanced language generation.

---

## Local Model Selection

### Recommended Models by Tier

| Tier | Model | Parameters | VRAM | Speed | Strengths |
|------|-------|------------|------|-------|-----------|
| **Tier 1** | Qwen2.5-1.5B-Instruct | 1.5B | 2GB | 100+ tok/s | Excellent instruction following |
| **Tier 1** | Phi-3.5-mini | 3.8B | 3GB | 80+ tok/s | Strong reasoning for size |
| **Tier 1** | SmolLM2-1.7B | 1.7B | 2GB | 100+ tok/s | Fast, good at classification |
| **Tier 2** | Qwen2.5-7B-Instruct | 7B | 6GB | 40+ tok/s | Best quality/speed ratio |
| **Tier 2** | Mistral-7B-Instruct | 7B | 6GB | 40+ tok/s | Strong reasoning |
| **Tier 2** | Llama-3.2-8B-Instruct | 8B | 7GB | 35+ tok/s | Good instruction following |
| **Tier 2** | Gemma-2-9B-Instruct | 9B | 8GB | 30+ tok/s | Strong multilingual |
| **Tier 2+** | Qwen2.5-14B-Instruct | 14B | 12GB | 25+ tok/s | Near cloud quality |

### Quantization for Speed

Using 4-bit quantization (GGUF/AWQ/GPTQ) reduces memory and increases speed:

| Model | FP16 VRAM | Q4 VRAM | Speed Gain |
|-------|-----------|---------|------------|
| Qwen2.5-7B | 14GB | 4.5GB | 2x |
| Mistral-7B | 14GB | 4.5GB | 2x |
| Qwen2.5-14B | 28GB | 9GB | 2x |

**Recommendation**: Run Q4_K_M quantization for best quality/speed tradeoff.

---

## Implementation Options

### Option 1: Ollama (Simplest)

**Pros**: Dead simple setup, built-in model management, REST API
**Cons**: Less control, overhead per request

```python
import ollama

class OllamaInference:
    def __init__(self, tier1_model="qwen2.5:1.5b", tier2_model="qwen2.5:7b"):
        self.tier1 = tier1_model
        self.tier2 = tier2_model
        
        # Pre-pull models
        ollama.pull(self.tier1)
        ollama.pull(self.tier2)
    
    async def decide_tier1(self, prompt: str) -> str:
        """Fast routine decision"""
        response = ollama.generate(
            model=self.tier1,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Low temp for consistency
                "num_predict": 50,   # Short responses
                "top_p": 0.9,
            }
        )
        return response['response']
    
    async def decide_tier2(self, prompt: str) -> str:
        """Contextual decision with more reasoning"""
        response = ollama.generate(
            model=self.tier2,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 200,
                "top_p": 0.95,
            }
        )
        return response['response']
```

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b

# Verify
ollama list
```

### Option 2: llama.cpp via llama-cpp-python (Best Performance)

**Pros**: Maximum speed, fine-grained control, batching support
**Cons**: More setup, manual model management

```python
from llama_cpp import Llama

class LlamaCppInference:
    def __init__(
        self,
        tier1_path: str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        tier2_path: str = "models/qwen2.5-7b-instruct-q4_k_m.gguf",
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_ctx: int = 2048,
    ):
        # Load tier 1 model (kept in memory)
        self.tier1 = Llama(
            model_path=tier1_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=512,
            verbose=False,
        )
        
        # Load tier 2 model
        self.tier2 = Llama(
            model_path=tier2_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=512,
            verbose=False,
        )
    
    def decide_tier1(self, prompt: str) -> str:
        """Synchronous tier 1 decision"""
        output = self.tier1.create_completion(
            prompt=prompt,
            max_tokens=50,
            temperature=0.1,
            top_p=0.9,
            stop=["\n", "Action:", "Decision:"],
        )
        return output['choices'][0]['text'].strip()
    
    def decide_tier2(self, prompt: str) -> str:
        """Synchronous tier 2 decision"""
        output = self.tier2.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
            top_p=0.95,
            stop=["\n\n", "---"],
        )
        return output['choices'][0]['text'].strip()
    
    def batch_decide_tier1(self, prompts: list[str]) -> list[str]:
        """Batch multiple tier 1 decisions for efficiency"""
        # llama.cpp supports batching for parallel inference
        results = []
        for prompt in prompts:
            results.append(self.decide_tier1(prompt))
        return results
```

**Setup**:
```bash
# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# Or with Metal (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Download models (using huggingface-cli)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/

huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models/
```

### Option 3: vLLM (Best for Multi-GPU/High Throughput)

**Pros**: Continuous batching, PagedAttention, OpenAI-compatible API
**Cons**: Heavier setup, requires more VRAM

```python
from openai import OpenAI

class VLLMInference:
    """Uses vLLM server with OpenAI-compatible API"""
    
    def __init__(
        self,
        tier1_url: str = "http://localhost:8001/v1",
        tier2_url: str = "http://localhost:8002/v1",
    ):
        self.tier1_client = OpenAI(base_url=tier1_url, api_key="dummy")
        self.tier2_client = OpenAI(base_url=tier2_url, api_key="dummy")
    
    async def decide_tier1(self, prompt: str) -> str:
        response = self.tier1_client.completions.create(
            model="qwen2.5-1.5b-instruct",
            prompt=prompt,
            max_tokens=50,
            temperature=0.1,
        )
        return response.choices[0].text.strip()
    
    async def decide_tier2(self, prompt: str) -> str:
        response = self.tier2_client.completions.create(
            model="qwen2.5-7b-instruct",
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].text.strip()
```

**Setup**:
```bash
pip install vllm

# Start tier 1 server
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.3

# Start tier 2 server (separate terminal)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8002 \
    --gpu-memory-utilization 0.6
```

### Option 4: Transformers (Most Flexible)

**Pros**: Full control, easy fine-tuning, familiar API
**Cons**: Slower than optimized runtimes

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class TransformersInference:
    def __init__(
        self,
        tier1_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        tier2_model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        # Load tier 1
        self.tier1_tokenizer = AutoTokenizer.from_pretrained(tier1_model)
        self.tier1_model = AutoModelForCausalLM.from_pretrained(
            tier1_model,
            torch_dtype=dtype,
            device_map=device,
        )
        
        # Load tier 2
        self.tier2_tokenizer = AutoTokenizer.from_pretrained(tier2_model)
        self.tier2_model = AutoModelForCausalLM.from_pretrained(
            tier2_model,
            torch_dtype=dtype,
            device_map=device,
        )
        
        # Create pipelines
        self.tier1_pipe = pipeline(
            "text-generation",
            model=self.tier1_model,
            tokenizer=self.tier1_tokenizer,
        )
        self.tier2_pipe = pipeline(
            "text-generation",
            model=self.tier2_model,
            tokenizer=self.tier2_tokenizer,
        )
    
    def decide_tier1(self, prompt: str) -> str:
        result = self.tier1_pipe(
            prompt,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tier1_tokenizer.eos_token_id,
        )
        return result[0]['generated_text'][len(prompt):].strip()
    
    def decide_tier2(self, prompt: str) -> str:
        result = self.tier2_pipe(
            prompt,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tier2_tokenizer.eos_token_id,
        )
        return result[0]['generated_text'][len(prompt):].strip()
```

---

## Prompt Engineering for Small Models

Small models require more structured prompts than large models. Key techniques:

### 1. Constrained Output Format

Force the model to respond in a specific format:

```python
TIER1_ACTION_PROMPT = """You are {agent_role}. Select your next action.

CURRENT STATE:
- Time: {time}
- Current task: {current_task}
- Task status: {task_status}
- Queue: {queue_summary}

AVAILABLE ACTIONS:
1. CONTINUE - Keep working on current task
2. COMPLETE - Mark current task as done
3. SWITCH - Switch to different task
4. BLOCKED - Report blocker
5. COMMUNICATE - Send message
6. IDLE - No action needed

OUTPUT FORMAT: Just the action number (1-6)

ACTION:"""

# Model outputs: "1" or "2" etc.
```

### 2. Few-Shot Examples

Include 2-3 examples in the prompt:

```python
TIER2_PRIORITY_PROMPT = """You are {agent_role}. Decide which task to work on.

EXAMPLE 1:
Tasks: [Bug fix (urgent, 2h), New feature (normal, 8h), Documentation (low, 4h)]
Deadline pressure: High
Decision: Bug fix - urgent priority overrides, and it's quick

EXAMPLE 2:
Tasks: [Review request (normal, 1h), Design work (normal, 6h)]
Deadline pressure: Low
Decision: Review request - unblocks colleague, quick win

YOUR SITUATION:
Tasks: {task_list}
Deadline pressure: {deadline_pressure}
Your expertise: {skills}

Decision:"""
```

### 3. Chain-of-Thought for Tier 2

For more complex decisions, guide the reasoning:

```python
TIER2_BLOCKER_PROMPT = """You are {agent_role}. Decide how to handle a blocker.

BLOCKER: {blocker_description}
TIME BLOCKED: {hours_blocked} hours
WHO CAN HELP: {potential_helpers}
DEADLINE: {deadline}

Think step by step:
1. Is this blocker critical for the deadline?
2. Have I waited a reasonable time for self-resolution?
3. Who is the best person to ask?
4. What's the cost of escalating vs waiting?

REASONING:
[Your step-by-step thinking]

DECISION: [WAIT/ESCALATE/WORKAROUND]
TARGET: [Person to contact if escalating]"""
```

### 4. Structured JSON Output

For complex decisions, use JSON:

```python
TIER2_TASK_PLAN_PROMPT = """You are {agent_role}. Plan your approach for this task.

TASK: {task_description}
ESTIMATED HOURS: {estimated_hours}
TOOLS AVAILABLE: {tools}
FILES NEEDED: {files}

Output a JSON plan:
{{
  "approach": "brief description of approach",
  "steps": ["step 1", "step 2", ...],
  "tools_needed": ["tool1", "tool2"],
  "potential_blockers": ["blocker1", ...],
  "estimated_hours": number
}}

JSON:"""
```

---

## Decision Router Implementation

The router determines which tier handles each decision:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import re

class DecisionTier(Enum):
    TIER1_LOCAL_SMALL = 1
    TIER2_LOCAL_MEDIUM = 2
    TIER3_CLOUD = 3

@dataclass
class DecisionContext:
    agent_id: str
    decision_type: str
    complexity_signals: dict
    requires_creativity: bool = False
    involves_multi_agent: bool = False
    is_novel_situation: bool = False

class DecisionRouter:
    """Routes decisions to appropriate LLM tier"""
    
    # Decision types that can be handled by Tier 1
    TIER1_DECISIONS = {
        "next_action",           # What to do next
        "task_transition",       # Moving between states
        "tool_selection",        # Which tool to use
        "time_estimate",         # How long will this take
        "accept_reject",         # Binary yes/no decisions
        "queue_position",        # What's next in queue
    }
    
    # Decision types requiring Tier 2
    TIER2_DECISIONS = {
        "task_prioritization",   # Multiple competing tasks
        "message_composition",   # Write a message
        "blocker_resolution",    # How to handle blockers
        "approach_selection",    # How to tackle a task
        "quality_assessment",    # Is this good enough
        "escalation_decision",   # Should I escalate
    }
    
    # Decision types requiring Tier 3 (cloud)
    TIER3_DECISIONS = {
        "conflict_resolution",   # Multi-agent conflicts
        "novel_problem",         # Never seen before
        "creative_solution",     # Needs creativity
        "complex_negotiation",   # Multi-party coordination
        "recovery_planning",     # Disaster recovery
    }
    
    def __init__(
        self,
        tier1_inference: "LocalInference",
        tier2_inference: "LocalInference", 
        tier3_inference: "CloudInference",
        cache: "DecisionCache",
    ):
        self.tier1 = tier1_inference
        self.tier2 = tier2_inference
        self.tier3 = tier3_inference
        self.cache = cache
        
    def route(self, context: DecisionContext) -> DecisionTier:
        """Determine which tier should handle this decision"""
        
        # Force tier 3 for certain conditions
        if context.involves_multi_agent and context.is_novel_situation:
            return DecisionTier.TIER3_CLOUD
        if context.requires_creativity:
            return DecisionTier.TIER3_CLOUD
            
        # Check decision type
        if context.decision_type in self.TIER1_DECISIONS:
            # Can we use cached response?
            if self.cache.has_similar(context):
                return DecisionTier.TIER1_LOCAL_SMALL
            return DecisionTier.TIER1_LOCAL_SMALL
            
        if context.decision_type in self.TIER2_DECISIONS:
            return DecisionTier.TIER2_LOCAL_MEDIUM
            
        if context.decision_type in self.TIER3_DECISIONS:
            return DecisionTier.TIER3_CLOUD
            
        # Default to tier 2 for unknown types
        return DecisionTier.TIER2_LOCAL_MEDIUM
    
    async def decide(self, context: DecisionContext, prompt: str) -> str:
        """Route and execute decision"""
        
        # Check cache first
        cached = self.cache.get(context, prompt)
        if cached:
            return cached
        
        # Route to appropriate tier
        tier = self.route(context)
        
        match tier:
            case DecisionTier.TIER1_LOCAL_SMALL:
                result = await self.tier1.decide(prompt)
            case DecisionTier.TIER2_LOCAL_MEDIUM:
                result = await self.tier2.decide(prompt)
            case DecisionTier.TIER3_CLOUD:
                result = await self.tier3.decide(prompt)
        
        # Cache result
        self.cache.store(context, prompt, result)
        
        return result
```

---

## Decision Caching

Many agent decisions are similar. Cache aggressively:

```python
import hashlib
from typing import Optional
from dataclasses import dataclass
import sqlite3
import json

@dataclass
class CacheEntry:
    context_hash: str
    prompt_hash: str
    response: str
    hit_count: int
    created_at: float

class DecisionCache:
    """LRU cache for LLM decisions with semantic similarity"""
    
    def __init__(
        self,
        db_path: str = "decision_cache.db",
        max_entries: int = 10000,
        similarity_threshold: float = 0.95,
    ):
        self.db_path = db_path
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                context_hash TEXT,
                prompt_hash TEXT,
                response TEXT,
                hit_count INTEGER DEFAULT 1,
                created_at REAL,
                PRIMARY KEY (context_hash, prompt_hash)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_context ON cache(context_hash)
        """)
        conn.commit()
        conn.close()
    
    def _hash_context(self, context: DecisionContext) -> str:
        """Create hash from context, ignoring volatile fields"""
        stable_context = {
            "agent_role": context.agent_id.split("-")[0],  # Role, not ID
            "decision_type": context.decision_type,
            # Normalize complexity signals
            "complexity_bucket": self._bucket_complexity(context.complexity_signals),
        }
        return hashlib.md5(json.dumps(stable_context, sort_keys=True).encode()).hexdigest()
    
    def _hash_prompt(self, prompt: str) -> str:
        """Hash prompt, normalizing variable parts"""
        # Remove timestamps, specific IDs, etc.
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', prompt)
        normalized = re.sub(r'\d{2}:\d{2}', 'TIME', normalized)
        normalized = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}', 'UUID', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _bucket_complexity(self, signals: dict) -> str:
        """Bucket complexity into categories for cache grouping"""
        # Simple heuristic bucketing
        if signals.get("queue_length", 0) > 5:
            return "high_load"
        if signals.get("deadline_days", 30) < 3:
            return "urgent"
        return "normal"
    
    def get(self, context: DecisionContext, prompt: str) -> Optional[str]:
        """Try to get cached response"""
        context_hash = self._hash_context(context)
        prompt_hash = self._hash_prompt(prompt)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT response FROM cache 
            WHERE context_hash = ? AND prompt_hash = ?
        """, (context_hash, prompt_hash))
        
        row = cursor.fetchone()
        if row:
            # Update hit count
            conn.execute("""
                UPDATE cache SET hit_count = hit_count + 1
                WHERE context_hash = ? AND prompt_hash = ?
            """, (context_hash, prompt_hash))
            conn.commit()
            conn.close()
            return row[0]
        
        conn.close()
        return None
    
    def store(self, context: DecisionContext, prompt: str, response: str):
        """Store decision in cache"""
        context_hash = self._hash_context(context)
        prompt_hash = self._hash_prompt(prompt)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO cache 
            (context_hash, prompt_hash, response, created_at)
            VALUES (?, ?, ?, ?)
        """, (context_hash, prompt_hash, response, time.time()))
        
        # Prune if over limit
        conn.execute("""
            DELETE FROM cache WHERE rowid IN (
                SELECT rowid FROM cache 
                ORDER BY hit_count ASC, created_at ASC
                LIMIT max(0, (SELECT COUNT(*) FROM cache) - ?)
            )
        """, (self.max_entries,))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_entries,
                SUM(hit_count) as total_hits,
                AVG(hit_count) as avg_hits
            FROM cache
        """)
        row = cursor.fetchone()
        conn.close()
        return {
            "total_entries": row[0],
            "total_hits": row[1],
            "avg_hits_per_entry": row[2],
        }
```

---

## Batching for Efficiency

Process multiple agent decisions in parallel:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchedInference:
    """Batch multiple agent decisions for efficient inference"""
    
    def __init__(
        self,
        inference: "LocalInference",
        batch_size: int = 8,
        batch_timeout: float = 0.1,  # seconds
    ):
        self.inference = inference
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.pending_requests: list = []
        self.pending_futures: list = []
        self.lock = asyncio.Lock()
        self.batch_event = asyncio.Event()
        
        # Start batch processor
        self._processor_task = None
    
    async def start(self):
        """Start the batch processor"""
        self._processor_task = asyncio.create_task(self._batch_processor())
    
    async def stop(self):
        """Stop the batch processor"""
        if self._processor_task:
            self._processor_task.cancel()
    
    async def decide(self, prompt: str) -> str:
        """Submit decision request, returns when batch is processed"""
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_requests.append(prompt)
            self.pending_futures.append(future)
            
            # Trigger batch if full
            if len(self.pending_requests) >= self.batch_size:
                self.batch_event.set()
        
        return await future
    
    async def _batch_processor(self):
        """Process batches of requests"""
        while True:
            # Wait for batch to fill or timeout
            try:
                await asyncio.wait_for(
                    self.batch_event.wait(),
                    timeout=self.batch_timeout
                )
            except asyncio.TimeoutError:
                pass
            
            # Get current batch
            async with self.lock:
                if not self.pending_requests:
                    self.batch_event.clear()
                    continue
                
                batch_prompts = self.pending_requests[:self.batch_size]
                batch_futures = self.pending_futures[:self.batch_size]
                
                self.pending_requests = self.pending_requests[self.batch_size:]
                self.pending_futures = self.pending_futures[self.batch_size:]
                self.batch_event.clear()
            
            # Process batch
            try:
                results = await self._process_batch(batch_prompts)
                
                # Resolve futures
                for future, result in zip(batch_futures, results):
                    future.set_result(result)
                    
            except Exception as e:
                # Reject all futures in batch
                for future in batch_futures:
                    future.set_exception(e)
    
    async def _process_batch(self, prompts: list[str]) -> list[str]:
        """Process a batch of prompts"""
        # For llama.cpp, process sequentially but benefit from KV cache
        # For vLLM, true batching is handled by the server
        results = []
        for prompt in prompts:
            result = self.inference.decide_tier1(prompt)
            results.append(result)
        return results
```

---

## Hybrid Architecture Integration

Complete integration with simulation engine:

```python
from dataclasses import dataclass
from typing import Protocol
import anthropic

class InferenceBackend(Protocol):
    async def decide(self, prompt: str) -> str: ...

@dataclass
class HybridLLMConfig:
    # Local model paths
    tier1_model: str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    tier2_model: str = "models/qwen2.5-7b-instruct-q4_k_m.gguf"
    
    # Cloud settings
    cloud_provider: str = "anthropic"
    cloud_model: str = "claude-3-haiku-20240307"
    cloud_api_key: str = ""
    
    # Routing thresholds
    tier3_budget_per_simulation: float = 10.0  # USD
    tier3_calls_per_agent_per_day: int = 2
    
    # Performance
    enable_batching: bool = True
    enable_caching: bool = True
    cache_db_path: str = "decision_cache.db"

class HybridLLMSystem:
    """Complete hybrid local/cloud LLM system"""
    
    def __init__(self, config: HybridLLMConfig):
        self.config = config
        
        # Initialize local models
        self.tier1 = LlamaCppInference(config.tier1_model)
        self.tier2 = LlamaCppInference(config.tier2_model)
        
        # Initialize cloud client
        self.cloud_client = anthropic.Anthropic(api_key=config.cloud_api_key)
        
        # Initialize cache
        self.cache = DecisionCache(config.cache_db_path) if config.enable_caching else None
        
        # Initialize router
        self.router = DecisionRouter(
            tier1_inference=self.tier1,
            tier2_inference=self.tier2,
            tier3_inference=self,
            cache=self.cache,
        )
        
        # Track cloud usage
        self.cloud_calls = 0
        self.cloud_cost = 0.0
        
        # Batching
        if config.enable_batching:
            self.tier1_batcher = BatchedInference(self.tier1)
            self.tier2_batcher = BatchedInference(self.tier2)
    
    async def start(self):
        """Start background processors"""
        if self.config.enable_batching:
            await self.tier1_batcher.start()
            await self.tier2_batcher.start()
    
    async def stop(self):
        """Stop background processors"""
        if self.config.enable_batching:
            await self.tier1_batcher.stop()
            await self.tier2_batcher.stop()
    
    async def agent_decide(
        self,
        agent: "EngineerAgent",
        decision_type: str,
        prompt: str,
        context: dict,
    ) -> str:
        """Main entry point for agent decisions"""
        
        # Build decision context
        decision_context = DecisionContext(
            agent_id=agent.id,
            decision_type=decision_type,
            complexity_signals=self._extract_complexity(context),
            requires_creativity=context.get("requires_creativity", False),
            involves_multi_agent=context.get("involves_multi_agent", False),
            is_novel_situation=context.get("is_novel", False),
        )
        
        # Route and execute
        result = await self.router.decide(decision_context, prompt)
        
        return result
    
    async def decide_cloud(self, prompt: str) -> str:
        """Cloud LLM call (Tier 3)"""
        
        # Check budget
        if self.cloud_cost >= self.config.tier3_budget_per_simulation:
            # Fallback to tier 2
            return await self.tier2.decide(prompt)
        
        # Make cloud call
        response = self.cloud_client.messages.create(
            model=self.config.cloud_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Track usage
        self.cloud_calls += 1
        input_cost = response.usage.input_tokens * 0.00025 / 1000
        output_cost = response.usage.output_tokens * 0.00125 / 1000
        self.cloud_cost += input_cost + output_cost
        
        return response.content[0].text
    
    def _extract_complexity(self, context: dict) -> dict:
        """Extract complexity signals from context"""
        return {
            "queue_length": len(context.get("queue", [])),
            "deadline_days": context.get("deadline_days", 30),
            "task_complexity": context.get("task_complexity", "medium"),
            "dependencies": len(context.get("dependencies", [])),
        }
    
    def get_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "cloud_calls": self.cloud_calls,
            "cloud_cost": self.cloud_cost,
            "cache_stats": self.cache.get_stats() if self.cache else None,
        }
```

---

## Performance Benchmarks

Expected performance on typical hardware:

### Consumer Hardware (RTX 3080/4080, 10-16GB VRAM)

| Model | Quantization | Tokens/sec | Decisions/sec |
|-------|--------------|------------|---------------|
| Qwen2.5-1.5B | Q4_K_M | 120 | 60 |
| Qwen2.5-7B | Q4_K_M | 45 | 15 |
| Qwen2.5-14B | Q4_K_M | 25 | 8 |

### Server Hardware (A100 40GB)

| Model | Quantization | Tokens/sec | Decisions/sec |
|-------|--------------|------------|---------------|
| Qwen2.5-1.5B | FP16 | 300+ | 150 |
| Qwen2.5-7B | FP16 | 150 | 50 |
| Qwen2.5-14B | FP16 | 80 | 25 |

### Simulation Throughput

With 50 agents, 15-min ticks, 6-month simulation:

| Config | Time to Complete | Cloud Cost |
|--------|------------------|------------|
| All Cloud (Claude) | N/A (too expensive) | ~$2,800 |
| All Tier 2 Local | ~45 minutes | $0 |
| Hybrid (95% local) | ~30 minutes | ~$15 |
| Hybrid + Cache | ~15 minutes | ~$8 |

---

## Fine-Tuning Considerations

For maximum accuracy, fine-tune on engineering-specific data:

### Training Data Sources

1. **Synthetic decisions**: Generate with Claude, validate with domain experts
2. **Historical project data**: If available from partner organizations
3. **Engineering workflow documentation**: SOPs, process guides
4. **Tool interaction logs**: RAPS CLI usage patterns

### Fine-Tuning Approach

```python
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# LoRA config for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Example training data format
training_examples = [
    {
        "prompt": "You are Senior Designer. Tasks: [Assembly review, Part modeling]. Priority?",
        "completion": "Assembly review - unblocks team, quick task"
    },
    {
        "prompt": "You are Junior Designer. Upload failed with error 416. Action?",
        "completion": "BLOCKED - report chunk size error to PLM Admin"
    },
    # ... thousands more
]

# Train
training_args = TrainingArguments(
    output_dir="./eddt-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
)

# Apply LoRA and train
model = get_peft_model(base_model, lora_config)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### Expected Improvement from Fine-Tuning

| Metric | Base Model | Fine-Tuned |
|--------|------------|------------|
| Action selection accuracy | 75% | 92% |
| Priority reasoning quality | 60% | 85% |
| Format compliance | 80% | 98% |
| Engineering terminology | 65% | 95% |

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT OPTIONS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  OPTION A: Single Machine (Development/Small Scale)                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GPU Server (RTX 4090 / A6000)                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ Tier 1 Model │  │ Tier 2 Model │  │  Simulation  │          │   │
│  │  │  (2GB VRAM)  │  │  (6GB VRAM)  │  │    Engine    │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  OPTION B: Containerized (Production)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Kubernetes Cluster                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │  Tier 1 Pod  │  │  Tier 2 Pod  │  │   Sim Pods   │          │   │
│  │  │  (vLLM x 2)  │  │  (vLLM x 1)  │  │   (x N)      │          │   │
│  │  │  2x T4 GPU   │  │  1x A10 GPU  │  │   CPU only   │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │            │               │                │                    │   │
│  │            └───────────────┴────────────────┘                    │   │
│  │                           │                                      │   │
│  │                    ┌──────┴──────┐                               │   │
│  │                    │   Redis     │  (Request queue + cache)      │   │
│  │                    └─────────────┘                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  OPTION C: Serverless (Burst Capacity)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ┌──────────────┐          ┌──────────────────────────┐        │   │
│  │  │ Local Tier 1 │          │  Modal / RunPod / Lambda │        │   │
│  │  │    Always    │──────────│  Tier 2 on demand        │        │   │
│  │  │     Hot      │          │  Scale to 0 when idle    │        │   │
│  │  └──────────────┘          └──────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Cost Comparison

| Approach | 6-Month Sim Cost | Speed | Quality |
|----------|------------------|-------|---------|
| 100% Claude Haiku | $2,800 | Slow | Excellent |
| 100% Local Qwen-7B | $0 (hardware only) | Fast | Good |
| Hybrid (95% local) | $15 | Fast | Very Good |
| Hybrid + Fine-tuned | $8 | Fast | Excellent |

**Recommendation**: Start with Hybrid approach using Ollama for simplicity, then optimize with llama.cpp and fine-tuning as needed.
