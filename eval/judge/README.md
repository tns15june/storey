# Judging claims against truth

Code scores what it can: generated sets, planted erasures and stair presence. A finding written in words, from Storey on an audited set or from the baseline anywhere, needs a judge to say which truth item it states. That judging was done blind:

1. `eval.report` writes one match sheet per set and method to `data/eval/sheets/`.
2. `blind.py` copies each sheet without its method, so a judge cannot tell Storey's claims from the baseline's, and keeps the key apart.
3. `workflow.js` (a Claude Code workflow) gives each sheet to two judges, Claude agents working separately, under the rules written at its top: the same problem at the same place, a general claim states nothing, and when unsure, no match. An adjudicating agent rules on every claim where the two differ.
4. The rulings become `eval/matches/<set>.<method>.json`, which `eval.report` reads back. Each mapping carries the sha256 of the sheet it answers, so a claim that changes afterwards makes the report refuse it.

On 1 November 2026 the judges ruled on 382 claims across 35 sheets and agreed on 371; the adjudicator ruled on the other 11. The Richmond mappings quote an examiner's letter and stay out of this public repository.
