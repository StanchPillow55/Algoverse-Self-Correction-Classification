# 🎯 Interview Prep - Algoverse Self-Correction Classification

**Last Updated:** 2026-04-29
**Documentation Pointers:** All claims link to internal docs in `docs/generated/`

---

## 60-90 Second STAR Pitch

### Situation
Research question: Do larger LLMs self-correct better than smaller ones? If so, what's the mathematical relationship between model size and improvement?

### Task
Build a comprehensive experimental pipeline to study self-correction scaling laws across 7 model sizes (1.8B-175B parameters) on 4 real benchmarks.

### Action
- Designed **Teacher-Learner architecture** with bias detection and multi-turn correction
- Built **multi-provider abstraction** for fair comparison (OpenAI, Anthropic, Replicate)
- Implemented **checkpointing system** for resumable experiments
- Created **ensemble voting system** with 4 aggregation strategies
- Developed **power-law analysis** pipeline for scaling law discovery

### Result

| Metric | Status | Evidence |
|--------|--------|----------|
| GPT-4o-mini GSM8K accuracy | **87.5%** (Confirmed) | `CURRENT_RESEARCH_STATUS.md` |
| GPT-4o-mini HumanEval accuracy | **82.3%** (Confirmed) | `CURRENT_RESEARCH_STATUS.md` |
| Total experimental runs | **119** (Confirmed) | Analysis output |
| Model sizes covered | **7** (1.8B-175B) | `README.md:7-8` |
| Datasets covered | **4** (GSM8K, HumanEval, SuperGLUE, MathBench) | `README.md:9` |

---

## Technical Deep Dive

### Architecture
- **Teacher-Learner Loop:** Bias detection → Template selection → Correction
- **Multi-Turn:** Up to 3 correction cycles per question
- **Multi-Provider:** OpenAI, Anthropic, Replicate with unified interface
- **Analysis:** Power-law fitting (Δ = A × ModelSize^α)

**Full details:** → `ARCHITECTURE.md`

### Key Tradeoffs

**Teacher-Learner vs Self-Prompting:**
- Chose separation for cleaner bias analysis
- Trade-off: More complex architecture

**Multi-Turn (max 3):**
- Research shows diminishing returns after 2-3 turns
- Trade-off: Cost vs. improvement data

**Deterministic Subsets:**
- Seeded random for reproducibility
- Trade-off: May not represent full distribution

**Full details:** → `DECISIONS.md`

---

## Drill-Down Q&A

### Q1: "What is the core research hypothesis?"

**Answer (Confirmed):** Larger LLMs should self-correct more effectively, following a power-law relationship: Δ = A × ModelSize^α, where Δ is accuracy improvement after correction.

**Evidence:** `README.md:456-464`

### Q2: "How does the Teacher detect bias?"

**Answer (Confirmed):** Four bias types detected:
1. **Overconfidence:** High confidence + wrong answer
2. **Underconfidence:** Low confidence + correct answer
3. **Pattern errors:** Systematic reasoning mistakes
4. **Calculation errors:** Arithmetic mistakes

**Evidence:** `src/agents/teacher.py`, `README.md:298-303`

### Q3: "Why multiple correction turns?"

**Answer (Confirmed):** Research shows improvement compounds across turns but with diminishing returns. 3 turns balances data richness with cost.

**Evidence:** `README.md:276-296`

### Q4: "How do you ensure reproducibility?"

**Answer (Confirmed):**
1. Deterministic dataset subsets with fixed seeds
2. Comprehensive trace logging (per-turn reasoning)
3. Checkpointing for resumable experiments
4. Environment variables (RUN_ID, DATASET_SPLIT, GIT_COMMIT)

**Evidence:** `README.md:453-454`, `CHECKPOINT_SYSTEM.md`

### Q5: "What's the checkpointing strategy?"

**Answer (Confirmed):**
- Atomic writes prevent corruption
- Checkpoint every N samples
- Resume from any checkpoint
- Auto-retry on API failures

**Evidence:** `CHECKPOINT_SYSTEM.md`

### Q6: "How do ensemble experiments work?"

**Answer (Confirmed):** Four voting strategies:
1. **Majority:** Most common answer wins
2. **Confidence-weighted:** Answers weighted by model confidence
3. **Consensus:** Require agreement threshold
4. **Adaptive:** Strategy varies by disagreement level

**Evidence:** `docs/ENSEMBLE_GUIDE.md`, `README.md:163-201`

### Q7: "What datasets do you use?"

**Answer (Confirmed):**
- **GSM8K (1000):** Math word problems
- **HumanEval (164):** Code generation with execution
- **SuperGLUE (1000):** Multi-task reasoning
- **MathBench (1000):** Advanced mathematics

All auto-downloaded from HuggingFace.

**Evidence:** `README.md:363-370`

### Q8: "How do you track costs?"

**Answer (Confirmed):**
- Model registry with per-token costs
- Real-time tracking during experiments
- Cost estimation before runs
- Power-law cost-benefit analysis

**Evidence:** `src/scaling/model_registry.py`, `README.md:309-332`

### Q9: "What were the key experimental findings?"

**Answer (Confirmed from completed runs):**
- GPT-4o-mini: 87.5% on GSM8K, 82.3% on HumanEval
- Claude-Haiku: 56.7% on GSM8K, 45.7% on HumanEval
- Multi-turn self-correction shows measurable accuracy gains
- Cost-benefit threshold identified at ~7B parameters

**Evidence:** `CURRENT_RESEARCH_STATUS.md:17-35`, `README.md:483-484`

### Q10: "What limitations exist?"

**Answer (Confirmed):**
- API quota constraints limit large-scale runs
- HumanEval code execution not fully sandboxed
- Full study costs ~$200
- Some experiments incomplete due to API failures

**Evidence:** `CURRENT_RESEARCH_STATUS.md:92-102`

---

## Reflection

### Technical Debt (Confirmed)

| Item | Issue |
|------|-------|
| API quota handling | Experiments blocked by rate limits |
| Llama integration | Replicate placeholder, not fully tested |
| SuperGLUE experiments | Incomplete due to API failures |

### What I'd Do Differently

| Change | Rationale |
|--------|-----------|
| Add caching layer | Reduce API costs for repeated queries |
| Better sandboxing | HumanEval code execution needs isolation |
| More model sizes | Better power-law fitting with more data points |

---

## Quick Reference Pointers

| Topic | Document |
|-------|----------|
| Architecture | `docs/generated/ARCHITECTURE.md` |
| Decisions/ADRs | `docs/generated/DECISIONS.md` |
| Testing | `docs/generated/TESTING.md` |
| Current status | `CURRENT_RESEARCH_STATUS.md` |
| Ensemble guide | `docs/ENSEMBLE_GUIDE.md` |
| ToolQA guide | `docs/TOOLQA_GUIDE.md` |
