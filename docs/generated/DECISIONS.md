# 📋 Decisions - Algoverse Self-Correction Classification

## ADR-001: Teacher-Learner Architecture

### Status
Accepted

### Context
Need to study self-correction in LLMs. Options:
1. Single model with self-prompting
2. Teacher-Learner with separate feedback generation
3. Ensemble voting without correction

### Decision
Teacher-Learner architecture where Teacher provides bias-aware feedback to Learner.

### Rationale
- Separates concerns: generation vs. evaluation
- Enables bias detection and targeted correction
- Supports multi-turn improvement tracking

### Evidence
`src/agents/learner.py`, `src/agents/teacher.py`

---

## ADR-002: Multi-Turn Correction Cycles

### Status
Accepted

### Decision
Support up to 3 correction turns per question with turn-by-turn reasoning traces.

### Rationale
- Research shows diminishing returns after 2-3 turns
- Allows measuring improvement delta over time
- Balances cost vs. data richness

### Evidence
`README.md:276-308`

---

## ADR-003: Deterministic Dataset Subsets

### Status
Accepted

### Context
Experiments must be reproducible across runs.

### Decision
Use seeded random sampling to create deterministic subsets (subset_20, subset_50, etc.).

### Rationale
- Same samples across runs for fair comparison
- Enables resumption from checkpoints
- Supports incremental study expansion

### Evidence
`src/data/scaling_datasets.py`

---

## ADR-004: Checkpointing System

### Status
Accepted

### Context
Experiments can take hours and cost hundreds of dollars. API failures common.

### Decision
Implement atomic checkpointing with resume capability.

### Rationale
- Prevents lost work from API failures
- Enables pause/resume for cost management
- Supports long-running experiments

### Evidence
`CHECKPOINT_SYSTEM.md`

---

## ADR-005: Multi-Provider Abstraction

### Status
Accepted

### Decision
Unified LearnerBot interface supporting OpenAI, Anthropic, and Replicate.

### Rationale
- Compare models across providers fairly
- Redundancy when one provider has issues
- Cover full parameter range (1.8B-175B)

### Evidence
`src/agents/learner.py`, `src/scaling/model_registry.py`

---

## ADR-006: Ensemble Voting System

### Status
Accepted

### Decision
Four voting strategies: majority, confidence-weighted, consensus, adaptive.

### Rationale
- Different strategies suit different tasks
- Enables cost-accuracy trade-offs
- Extends single-model study

### Evidence
`docs/ENSEMBLE_GUIDE.md`, `src/ensemble/voting.py`

---

## ADR-007: Power-Law Analysis

### Status
Accepted

### Context
Need to quantify relationship between model size and self-correction ability.

### Decision
Fit power-law: Δ = A × ModelSize^α to improvement data.

### Rationale
- Standard approach in scaling law research
- Provides interpretable scaling exponent
- Enables cost-benefit threshold identification

### Evidence
`src/scaling/analysis.py`

---

## Decision Log

| ID | Decision | Status |
|----|----------|--------|
| ADR-001 | Teacher-Learner architecture | Accepted |
| ADR-002 | Multi-turn correction (max 3) | Accepted |
| ADR-003 | Deterministic subsets | Accepted |
| ADR-004 | Checkpointing system | Accepted |
| ADR-005 | Multi-provider abstraction | Accepted |
| ADR-006 | Ensemble voting system | Accepted |
| ADR-007 | Power-law analysis | Accepted |
