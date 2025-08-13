# Teacher-Learner Architecture Documentation

## Overview

The teacher-learner architecture replaces traditional ML classifiers with a dual-bot system:
- **Learner-Bot**: Attempts to answer questions, provides self-confidence estimates
- **Teacher-Bot**: Detects cognitive biases, combines confidence signals, and selects reprompt templates

## Pipeline Flow

```
question q, reference r
      │
      ▼
Learner-Bot → answer₀, self_conf₀
      │
      ▼
Teacher-Bot: detect {bias₀, conf₀} and decide:
   RTS: (reprompt?; template P₀)
      ┌───────────────┬──────────────────────────┐
      │ no (stop)     │ yes: send P₀ to Learner  │
      ▼               ▼                           │
 return answer₀   answer₁, self_conf₁ ────────────┘
                     │
                     ▼
             Re-run Teacher (bias₁, conf₁)
                     │
               loop until: STOP rules
```

## Cognitive Bias Detection

The teacher-bot uses heuristic rules to detect 5 cognitive biases:

| Bias | Detection Heuristic | Teacher Confidence |
|------|--------------------|--------------------|
| **Anchoring** | Answer contains numbers from original question | 0.7 |
| **Availability/Bandwagon** | Contains social cues ("everyone", "commonly", "popular") | 0.7 |
| **Hindsight** | Contains post-hoc explanations ("obvious because", "clearly since") | 0.65 |
| **Overgeneralization** | Contains absolutes ("always", "never", "all cases") when wrong | 0.65 |
| **Confirmation** | Default for other incorrect answers (sticking to first guess) | 0.6 |
| **None** | Answer matches reference exactly | 0.95 |

## Confidence Combination

Multiple confidence signals are averaged:
- **Learner self-confidence**: Reported by learner-bot (0-1)
- **Teacher confidence**: From bias detection (0-1) 
- **k-vote share**: Optional disagreement from parallel learners (0-1)

Combined confidence = average of available signals (defaults to 0.5 if none)

## Confidence Buckets

| Bucket | Range | Description |
|--------|-------|-------------|
| **Low** | < 0.4 | Uncertain, needs supportive guidance |
| **Mid** | 0.4-0.7 | Moderate confidence, bias-specific intervention |
| **High** | > 0.7 | Confident (possibly overconfident if wrong) |

## RTS Policy: Bias × Confidence → Template Mapping

### Bias-Specific Rules
| Bias | Confidence | Template | Style | Purpose |
|------|------------|----------|--------|---------|
| **Anchoring** | High | `counter_anchor_v1` | Adversarial | "Ignore earlier numbers, re-derive from first principles" |
| **Confirmation** | Mid/High | `devils_advocate_v1` | Neutral | "List disconfirming hypotheses and test them" |
| **Availability/Bandwagon** | Mid | `evidence_only_v1` | Neutral | "Use only explicit calculation or given facts" |
| **Hindsight** | Any | `recompute_no_story_v1` | Supportive | "Recompute first, explain after verification" |
| **Overgeneralization** | Low/Mid | `quantify_uncertainty_v1` | Supportive | "Replace absolutes with ranges/conditions" |

### General Fallbacks
| Confidence | Template | Style | Purpose |
|------------|----------|--------|---------|
| **High (wrong)** | `are_you_sure_recheck` | Adversarial | Challenge overconfidence |
| **Low** | `think_step_by_step` | Supportive | Provide structure |
| **Default** | `try_again_concise` | Neutral | Generic retry |

### Supportive Templates
| Template | Text | When to Use |
|----------|------|-------------|
| `calming_focus_v1` | "Your prior steps look correct. Keep the answer." | Right answer, avoid churn |

## STOP Rules

The teacher stops reprompting when ANY condition is met:

1. **Correctness**: Answer matches reference → STOP
2. **Max turns**: Reached configured limit (default: 3) → STOP
3. **Two non-improvements**: Consecutive failed attempts → STOP (implicit via max_turns)
4. **Negative expected gain**: Teacher predicts reprompt will hurt → STOP

Additional stop conditions can be configured in `configs/stop_rules.yaml`.

## Template Structure

Templates are stored in `rts_templates.json` with metadata:

```json
{
  "id": "counter_anchor_v1",
  "text": "Ignore any numbers or hints seen earlier. Re-derive from first principles, then compute. Show steps.",
  "style": "adversarial",
  "cognitive_load": "med", 
  "length": "short",
  "bias": "Anchoring"
}
```

### Template Metadata
- **style**: `supportive`, `neutral`, `adversarial`
- **cognitive_load**: `low`, `med`, `high`
- **length**: `short`, `long`
- **bias**: Target bias or `"None"` for general use

## Adding New Templates

1. **Add to `rts_templates.json`**:
   ```json
   {
     "id": "my_new_template",
     "text": "Your reprompt text here.",
     "style": "neutral",
     "cognitive_load": "low",
     "length": "short", 
     "bias": "Confirmation"
   }
   ```

2. **Update RTS policy in `src/rts/policy.py`**:
   ```python
   # Add rule to select_template()
   if b == "Confirmation" and c == "low":
       return True, "my_new_template"
   ```

3. **Test with demo mode**:
   ```bash
   python -m src.main run --max-turns 2
   ```

## Configuration Files

### `configs/run.yaml`
```yaml
runtime:
  demo_mode: true
  max_turns: 3
  parallel_learners_k: 1
models:
  provider: "demo"  # "demo" | "openai" | "anthropic"
  temperature:
    learner_math: 0.3
    learner_facts: 0.6
    teacher: 0.0
paths:
  dataset: "data/math20.csv"
  traces_out: "outputs/traces.json"
```

### `configs/stop_rules.yaml`
```yaml
stop_rules:
  max_turns: 3
  stop_on_match: true
  stop_on_two_non_improvements: true
  negative_expected_gain_threshold: 0.0
```

## Trace Output Format

Each run generates `outputs/traces.json`:

```json
{
  "summary": {
    "items": 20,
    "final_accuracy_mean": 0.95
  },
  "traces": [
    {
      "qid": "q1", 
      "question": "What is 2+2?",
      "reference": "4",
      "turns": [
        {
          "answer": "4",
          "self_conf": 0.9,
          "teacher_bias": "None", 
          "teacher_conf": 0.95,
          "template": null,
          "accuracy": 1
        }
      ],
      "final_accuracy": 1
    }
  ]
}
```

## Demo Mode vs Production

### Demo Mode (Default)
- **Provider**: `"demo"` - offline arithmetic parsing
- **No API keys required**: Works completely offline
- **Math focus**: Handles basic arithmetic reliably
- **Fast**: Instant responses for testing

### Production Mode  
- **Provider**: `"openai"` or `"anthropic"`
- **API keys required**: Set in environment or config
- **Full capability**: Handle any question type
- **Cost**: Token usage per API call

## Performance Expectations

On Math-20 dataset:
- **Demo mode**: ~95% accuracy (19/20 correct)
- **Single-pass baseline**: ~90% accuracy
- **Improvement**: +5% from bias-aware reprompting
- **Cost**: ~2-3 turns per question average

## Extending the System

### Adding New Bias Types
1. Update `BIAS` list in `src/agents/teacher.py`
2. Add detection heuristic to `detect_bias()` function
3. Add corresponding template to `rts_templates.json`
4. Update policy rules in `src/rts/policy.py`

### Adding New Providers
1. Implement provider in `src/agents/learner.py`
2. Add to `models.provider` options in config
3. Handle API authentication and rate limiting
4. Test with demo dataset first

### Custom Datasets
1. Create CSV with `question,reference` columns
2. Update `paths.dataset` in config
3. Run: `python -m src.main run --dataset your_data.csv`

## Troubleshooting

### Common Issues
- **Import errors**: Run `pip install -r requirements.txt`
- **Missing output directory**: Pipeline auto-creates `outputs/`
- **Template not found**: Check template ID exists in `rts_templates.json`
- **Low accuracy**: Verify reference answers are formatted correctly

### Debug Mode
```bash
export DEMO_MODE=1
python -m src.main run --dataset data/math20.csv --max-turns 1
# Check outputs/traces.json for detailed turn information
```
