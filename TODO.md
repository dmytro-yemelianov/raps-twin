# TODO

- Convert `DecisionCache` to true async using `aiosqlite` (done).
- Replace event queue with `heapq` min-heap and remove `PriorityQueue` internals (done).
- Offload synchronous llama.cpp calls from the event loop (done).
- Add structured logging with JSON formatter and config via env (done).
- Align README with both `/health` and `/api/v1/health` endpoints (done).
- Add more unit tests:
  - Deterministic runs across multiple agents and tasks.
  - Tool layer failure paths recorded in metrics/logs.
  - Router tier selection edge cases.
- Introduce `aiosqlite` connection pooling or shared connection for performance.
- Add CORS/authn for API if exposed beyond local/dev.
- Add CI: lint, type-check, tests, coverage gate.
- Expose metrics/health via `/metrics` (Prometheus) and `/readyz`.
- Persist simulations and metrics in Postgres; replace in-memory storage.
- Implement graceful shutdown: flush metrics, close HTTP clients, DB.
- Add richer agent prompts and output parsers; reduce randomness.
