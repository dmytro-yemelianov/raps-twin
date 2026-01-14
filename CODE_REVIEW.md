# Code Review: EDDT (raps-twin)

## 🔴 Critical Issues

### 1. Dual State Storage Causes Data Inconsistency
**Location:** `src/eddt/api/routes.py:22`, `68-80`

```python
# In-memory storage (compat view over manager)
simulations: Dict[str, Dict] = {}
```
The routes maintain a separate `simulations` dict that can get out of sync with `sim_manager._sims`. `list_simulations()` reads from `sim_manager`, but `get_simulation()` reads from the local dict. If an external create happens, `get_simulation()` will 404.

### 2. Unbounded Memory Growth in MetricsCollector
**Location:** `src/eddt/simulation/metrics.py:36-43`

```python
self.agent_actions: List[Dict] = []
self.state_transitions: List[StateTransition] = []
```
All actions and transitions are stored in lists that grow indefinitely. A long simulation (365 days × many agents × ticks every 15 min) will consume gigabytes of memory.

### 3. Blocking LLM Inference in Async Context
**Location:** `src/eddt/llm/llama_cpp_client.py:50-64`

```python
output = await loop.run_in_executor(None, _call)
```
Using `None` as executor means the default thread pool (typically 5 threads). With many agents making concurrent LLM calls, this creates a bottleneck. Additionally, `llama.cpp` is not thread-safe by default—multiple concurrent calls to the same `Llama` instance can cause race conditions.

---

## 🟠 High-Priority Issues

### 4. Race Condition in Simulation State Updates
**Location:** `src/eddt/api/sim_manager.py:86-112`

```python
async def updater():
    while True:
        record["current_time"] = engine.simulation_time
        record["metrics"] = engine.metrics.compile()
        record["agent_states"] = [...]
        await asyncio.sleep(0.25)
```
The updater reads mutable state from the engine while the simulation is running in another coroutine. No locking mechanism protects these reads, risking torn reads/inconsistent snapshots.

### 5. MD5 Hash for Caching (Weak & Deprecated)
**Location:** `src/eddt/llm/cache.py:94, 104`

```python
return hashlib.md5(...).hexdigest()
```
MD5 is cryptographically broken and has collision vulnerabilities. While not a security issue here, it's considered deprecated. Use SHA-256 for better collision resistance.

### 6. Infinite Loop Risk in TUI observe_server
**Location:** `src/eddt/tui/dashboard.py:111-160`

```python
while True:
    # No exit condition except status == "completed"
```
If the simulation errors out or the server crashes, this loop runs forever polling. No timeout, no error handling for permanent failures, no graceful shutdown.

### 7. Hardcoded 40-Hour Week in Bottleneck Detection
**Location:** `src/eddt/simulation/metrics.py:141`

```python
utilization = self.calculate_agent_utilization(agent_id, 40.0)  # Assume 40h week
```
The simulation length varies, but bottleneck detection always assumes 40 hours. A 4-hour simulation will report wildly incorrect utilization percentages.

---

## 🟡 Medium-Priority Issues

### 8. Silent Exception Swallowing
**Location:** `src/eddt/api/sim_manager.py:132-136`, `src/eddt/agents/engineer.py:228-235`

```python
try:
    if engine:
        engine.stop()
except Exception:
    pass
```
Exceptions are silently swallowed. This hides real errors like "engine in invalid state" that should be logged.

### 9. State Machine Callback Exception Handling Too Broad
**Location:** `src/eddt/agents/state_machine.py:72-80`

```python
try:
    from_state = getattr(event_data.transition.source, "name", str(event_data.state))
    ...
except Exception:
    from_state = str(self.state)
    ...
```
Catching all `Exception` types masks real bugs. The callback should log what went wrong.

### 10. Random Import Inside Method
**Location:** `src/eddt/agents/engineer.py:183-185`

```python
def _should_block(self) -> bool:
    import random
    return random.random() < 0.1
```
Import at module level for clarity and slight performance benefit. Also, using global `random` instead of engine's seeded random breaks determinism.

### 11. `datetime.now()` Breaks Determinism
**Location:** `src/eddt/simulation/metrics.py:149, 155, 163, 179`

```python
timestamp=datetime.now()
```
`detect_bottlenecks()` uses real wall-clock time instead of simulation time, making metrics non-reproducible and comparing incompatible timestamps.

### 12. No Connection Cleanup in DecisionCache
**Location:** `src/eddt/llm/cache.py` (entire class)

The `DecisionCache` class stores a connection but has no `close()` method. For long-running services, this prevents proper cleanup.

### 13. PriorityQueue Not Async-Safe
**Location:** `src/eddt/agents/base.py:57`

```python
self.task_queue = PriorityQueue()
```
`queue.PriorityQueue` is not asyncio-aware. Multiple async tasks accessing it without locks can cause issues.

---

## 🟢 Low-Priority / Code Quality Issues

### 14. Type Hints Using `any`
**Location:** `src/eddt/simulation/environment.py:71`

```python
self.agents: Dict[str, any] = {}
```
Should be `Any` from typing, not `any` (which is a builtin function).

### 15. Unused `similarity_threshold` Parameter
**Location:** `src/eddt/llm/cache.py:34, 45`

```python
similarity_threshold: float = 0.95,  # not used in current implementation
```
The docstring even admits it's unused. Dead code should be removed or implemented.

### 16. Inconsistent Private Method Naming
**Location:** `src/eddt/api/sim_manager.py:141`

```python
def _public(self, sim_id: str) -> Dict:
```
This is called internally but returns public data. The name is confusing.

### 17. Magic String "CONTINUE"
**Location:** `src/eddt/api/sim_manager.py:22-24`

```python
async def decide(self, ...):
    return "CONTINUE"
```
Mock inference always returns "CONTINUE" but nothing parses this value. The LLM response is essentially ignored.

### 18. Dashboard Creates Classes Inside Loop
**Location:** `src/eddt/tui/dashboard.py:129-147`

```python
class _A:
    def __init__(self, d):
        ...
class _M:
    def compile(self):
        ...
```
Defining classes inside a polling loop is inefficient. Move them to module level.

---

## 🏗️ Architecture Concerns

### 19. No Graceful Shutdown
The simulation engine and manager have no cleanup lifecycle. Running background tasks (`updater_task`) are cancelled abruptly without cleanup.

### 20. Global Singleton Pattern
**Location:** `src/eddt/api/sim_manager.py:156`

```python
sim_manager = SimulationManager()
```
Module-level singleton makes testing difficult and prevents running multiple isolated instances.

### 21. Mixed Sync/Async Patterns
The codebase mixes synchronous `PriorityQueue` with async code. Consider using `asyncio.PriorityQueue` for consistency.

### 22. No Rate Limiting on API
The FastAPI routes have no rate limiting. A client could create thousands of simulations and exhaust memory.

---

## Summary Table

| Severity | Count | Main Areas |
|----------|-------|------------|
| Critical | 3 | State sync, memory, thread safety |
| High | 4 | Race conditions, infinite loops, incorrect calculations |
| Medium | 6 | Exception handling, determinism, resource cleanup |
| Low | 5 | Code quality, typing, naming |

---

## Fix Status

- [x] Issue 1: Dual State Storage - Removed local `simulations` dict, now uses `sim_manager` exclusively
- [x] Issue 2: Unbounded Memory Growth - Added `max_actions` and `max_transitions` limits with trimming
- [x] Issue 3: Blocking LLM Inference - Added threading.Lock and dedicated ThreadPoolExecutor
- [x] Issue 4: Race Condition in State Updates - Added per-simulation asyncio.Lock
- [x] Issue 5: MD5 Hash - Replaced with SHA-256
- [x] Issue 6: Infinite Loop in TUI - Added timeout, max_consecutive_errors, proper exit conditions
- [x] Issue 7: Hardcoded 40-Hour Week - Now calculates from actual simulation duration
- [x] Issue 8: Silent Exception Swallowing - Added logging.warning calls
- [x] Issue 9: Broad Exception Handling - Added logging.warning calls
- [x] Issue 10: Random Import Inside Method - Moved import to module level
- [x] Issue 11: datetime.now() Breaks Determinism - Added current_time parameter to detect_bottlenecks()
- [x] Issue 12: No Connection Cleanup - Added close() method and async context manager
- [ ] Issue 13: PriorityQueue Not Async-Safe - *Deferred: requires significant refactor*
- [x] Issue 14: Type Hints Using `any` - Changed to `Any` from typing
- [x] Issue 15: Unused similarity_threshold - Removed parameter
- [x] Issue 16: Inconsistent Naming - Renamed `_public` to `_to_public_view`
- [ ] Issue 17: Magic String "CONTINUE" - *Deferred: mock behavior, not production issue*
- [x] Issue 18: Classes Inside Loop - Moved `_AgentView` and `_MetricsView` to module level

### Architecture Concerns

- [x] Issue 19: No Graceful Shutdown - Added `shutdown()` method and FastAPI shutdown hook
- [x] Issue 20: Global Singleton Pattern - Made `sim_manager` replaceable, added async context manager
- [ ] Issue 21: Mixed Sync/Async Patterns - *Partially addressed with AsyncPriorityQueue*
- [x] Issue 22: No Rate Limiting on API - Added `RateLimitMiddleware` (60 req/min, burst 10)
