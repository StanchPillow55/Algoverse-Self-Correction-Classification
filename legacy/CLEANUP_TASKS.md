# Future Cleanup Tasks

## Code Organization
- [ ] Remove legacy classifier components after teacher-learner is proven
- [ ] Consolidate `src/` directory structure
- [ ] Remove unused imports and dependencies
- [ ] Standardize naming conventions

## Files to Review for Removal (AFTER PIVOT WORKS)
- [ ] `src/classifier/` (multi-head transformer classifier)
- [ ] `src/logits_features.py`
- [ ] `src/feature_fusion.py` 
- [ ] Related test files in `tests/`
- [ ] Legacy training data and scripts

## Documentation Updates
- [ ] Update README.md to reflect new architecture
- [ ] Remove classifier-specific documentation
- [ ] Add teacher-learner pipeline documentation

## Dependencies
- [ ] Remove ML/PyTorch dependencies if no longer needed
- [ ] Add OpenAI API dependencies
- [ ] Clean up requirements.txt

---
**Note: DO NOT START THESE TASKS until teacher-learner pipeline is fully working!**
