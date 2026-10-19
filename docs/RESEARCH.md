# Research plan

## Claim
A small, locally-embedded RAG system with an 8B generator can be made more faithful, cheaper to verify and better calibrated by (1) fusing lexical and dense ranking at chunk level, (2) verifying answer sentences with an embedding-first cascade, (3) repairing only unsupported sentences, and (4) treating the self-evaluation score as a calibrated confidence for abstention.

## Experimental protocol
1. **Data.** Do not rely on your own few documents. Use public sets with gold answers and passages, for example SQuAD 2.0 (includes unanswerable questions, which test abstention), HotpotQA (multi-hop) and Natural Questions. Take 200 to 500 questions per set; index their source passages as documents.
2. **Judge.** Keep `JUDGE_MODEL` (default `openai/gpt-oss-120b`) different from and stronger than `GEN_MODEL` (default `openai/gpt-oss-20b`). Report the agreement between the judge and token F1 or exact match on a 50-question hand-labelled sample.
3. **Ablation.** `scripts/run_ablation.py` runs A0 to A5; each row adds one component. Report accuracy with Wilson 95% intervals, faithfulness, ECE, AURC, LLM calls and latency. Paired bootstrap p-values compare each row with the previous one.
4. **Baselines to add.** A plain dense-only RAG (A0 already), BM25-only, and a published Self-RAG or CRAG variant if you can run one.
5. **Cascade analysis.** Sweep `hi` and `lo` in `research.verify_answer` and plot verdict agreement with the all-LLM verifier against LLM calls saved. This is likely your most original figure.
6. **Calibration.** Report reliability diagrams and risk–coverage curves for raw versus recalibrated confidence on a held-out split.

## Threats to validity (write these in the paper)
- A generator judging its own output is biased toward its own style.
- The system-side verifier and the measurement verifier must differ; they do here (cascade vs `llm_only`).
- Hand-set thresholds (`hi`=0.65, `lo`=0.30, retrieval top-k, RRF k=60) must be tuned on a dev split and reported on a separate test split.
- With fewer than about 50 questions per configuration the intervals are too wide to support claims.
- MiniLM similarity is weak for numbers, negation and multi-sentence reasoning; list failure cases.
