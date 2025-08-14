# Pivot Ground Rules: Teacher-Learner Architecture

## Development Philosophy
**"FUNCTIONALITY FIRST; CLEANUP LAST"**

## Priority Order
1. **Get the new teacher-bot/learner-bot pipeline working end-to-end**
2. Implement core components (bias detection, RTS, multi-pass loop)
3. Generate traces and validate basic functionality
4. ONLY THEN: cleanup legacy code

## What NOT to Do (Yet)
- ❌ Do NOT delete or refactor legacy classifier code
- ❌ Do NOT remove existing tests or data files
- ❌ Do NOT worry about code organization/cleanup during development
- ❌ Do NOT optimize for performance until functionality works

## What TO Do
- ✅ PRIORITIZE getting the pipeline working end-to-end
- ✅ Add new code alongside existing code
- ✅ Keep legacy components intact for reference
- ✅ Focus on proving the concept works
- ✅ Test incrementally as we build

## Branch Structure
```
pivot/teacher-learner-rts/
├── (existing legacy code - UNTOUCHED)
├── src/teacher_learner/     # New architecture
├── tests/teacher_learner/   # New tests
├── data/teacher_learner/    # New datasets
└── CLEANUP_TASKS.md         # Future cleanup tasks
```

## Development Checkpoints
After each major component:
1. Run smoke tests: `pytest -q` or `pytest tests/smoke`
2. Commit with clear messages
3. Document what works/doesn't work

## Final Cleanup Phase (LAST STEP)
- Remove deprecated classifier code
- Consolidate directories
- Clean up imports and dependencies
- Optimize structure

---
**Remember: We're proving a concept first, optimizing second!**
