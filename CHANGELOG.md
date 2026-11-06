# Changelog
## v2.0 (2026-11-06)
- Added Step 6 (LLM-judge baselines, second generator, claim-free consensus, SciFact, cost timing, SciFact paired re-judge), Step 7 (threshold, transfer, contradiction-feature analyses), Step 8 (hybrid and cascade analyses) data, scripts and plans.
- Step 8 deviation D1: comparators initially used clipped probabilities, which created ties; re-run with raw probabilities. Conclusions changed (hybrids do not beat the judge) and are reported as corrected.
- Added `reproduce_key_numbers.py`; made main analysis scripts repo-relative.
- Added S14 classification of the 53 low-C passages.
## v1.0
- Initial release (Steps 1-5).
