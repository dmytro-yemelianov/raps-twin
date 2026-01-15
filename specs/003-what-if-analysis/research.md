# Research: What-If Analysis

**Feature**: 003-what-if-analysis
**Date**: 2026-01-15

## Research Questions Resolved

### 1. How to parse modification syntax?

**Decision**: Support three syntax forms via regex:
1. Structured: `+1 senior_designer`, `-2 junior_designer`, `+50% part_design`
2. Natural language patterns: "add another senior designer", "double the reviews"
3. Explicit dict for programmatic use

**Rationale**: Regex handles common patterns without NLP dependencies. Pattern library covers 80% of use cases per spec requirement SC-002.

**Alternatives Considered**:
- Full NLP (spaCy, transformers): Rejected - adds large dependencies, violates simplicity principle
- Structured only: Rejected - reduces usability for non-technical users
- LLM parsing: Rejected - non-deterministic, requires network

### 2. How to safely modify configurations?

**Decision**: Deep copy baseline config dict, apply modifications in order, validate result before running.

**Rationale**: Python's `copy.deepcopy()` ensures baseline is never mutated. Validation catches impossible states (negative agents, zero tasks) before simulation.

**Alternatives Considered**:
- Modify in place with rollback: Rejected - complex, error-prone
- Create new YAML file: Rejected - unnecessary I/O, file clutter

### 3. How to handle conflicting modifications?

**Decision**: Apply modifications in specified order. Final state is what runs. Log each modification step.

**Rationale**: Sequential application is predictable and debuggable. Users can see exactly what happened.

**Alternatives Considered**:
- Reject conflicting modifications: Rejected - too restrictive
- Merge modifications: Rejected - undefined behavior for contradictions

### 4. What natural language patterns to support?

**Decision**: Pattern library for common phrases:
- "add another {role}" → `+1 {role}`
- "remove a {role}" → `-1 {role}`
- "double the {task_type}" → `+100% {task_type}`
- "halve the {task_type}" → `-50% {task_type}`
- "increase {task_type} by N" → `+N {task_type}`

**Rationale**: These cover ~80% of expected queries based on spec user stories.

**Alternatives Considered**:
- Comprehensive NLP: Rejected - overkill for known use cases
- No natural language: Rejected - reduces accessibility

### 5. How to present comparison results?

**Decision**: Integrate with feature 001 (Scenario Comparison) if available. Otherwise, include minimal side-by-side comparison in this module.

**Rationale**: Avoid duplication. Feature 001 handles comparison comprehensively.

**Alternatives Considered**:
- Always include full comparison logic: Rejected - duplicates 001
- Require 001 as dependency: Rejected - should work standalone

## Best Practices Applied

### Configuration Manipulation

- Use `copy.deepcopy()` for safe copying
- Validate after modification, not during
- Preserve original for comparison

### Pattern Matching

- Compile regexes once at module load
- Order patterns from most specific to least
- Include catch-all that requests clarification

### Error Messages

- Show what was attempted and why it failed
- Suggest correct syntax when pattern not recognized
- Include current config state in validation errors

## Technology Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Parsing approach | Regex patterns | No external dependencies, deterministic |
| Config manipulation | `copy.deepcopy()` | Safe, simple, stdlib |
| Module location | New `eddt/whatif.py` | Single responsibility |
| Comparison | Delegate to feature 001 or minimal inline | Avoid duplication |

## Natural Language Pattern Library

```python
PATTERNS = [
    # Agent modifications
    (r"add (?:another|one more|a) (\w+)", lambda m: f"+1 {m.group(1)}"),
    (r"remove (?:a|one) (\w+)", lambda m: f"-1 {m.group(1)}"),
    (r"add (\d+) (?:more )?(\w+)", lambda m: f"+{m.group(1)} {m.group(2)}"),

    # Task modifications
    (r"double (?:the )?(\w+)", lambda m: f"+100% {m.group(1)}"),
    (r"halve (?:the )?(\w+)", lambda m: f"-50% {m.group(1)}"),
    (r"increase (\w+) by (\d+)", lambda m: f"+{m.group(2)} {m.group(1)}"),
    (r"reduce (\w+) by (\d+)%?", lambda m: f"-{m.group(2)}% {m.group(1)}"),
]
```

## Dependencies

No new dependencies required. Uses:
- `re` (stdlib) for pattern matching
- `copy` (stdlib) for deepcopy
- Existing EDDT model and config infrastructure
- Optionally feature 001 for comparison
