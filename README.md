# Update Adherence Benchmark

Deterministic benchmark for **Update Adherence (UA)** and behavioral compliance in LLM memory agents. This repository is the companion artifact of the manuscript *"A Deterministic Benchmark for Store-Level and Behavior-Level Update Adherence in Large Language Model Memory Agents"* (under review at IEEE Access).

UA measures whether a memory-augmented agent acts on a revised constraint rather than the version it replaced. Scoring is fully programmatic at two layers, the assembled context (store level) and the generated answer (behavior level), with no LLM judge.

## Layout

| Path | Content |
| --- | --- |
| `engine/` | Frozen evaluation engine (memory architectures, scoring). Identical in every executable statement to the engine used for the reported runs — release copies differ only in comment cleanup; SHA-256 digests in `MANIFEST.sha256`. |
| `scenarios/holdout_v4/` | Batch A, 36 scenarios (12 Update Conflict, 12 Forgetting, 12 Noise). |
| `scenarios/holdout_v5/` | Batch B, 36 scenarios. |
| `scenarios/implicit/` | Implicit-revision variants of the 12 Batch-A Update Conflict scenarios (12 files; one of three surface forms rotated across the scenarios). |
| `scenarios/fn/` | Forgetting and Noise subset used by the 0.85-threshold regression check. |
| `generator/` | Scenario generator and templates. |
| `run_holdout_v4.py`, `run_holdout_v5.py` | Canonical benchmark runs (recall, CCR, store-level UA). |
| `run_behavior_probe.py` | Behavior-level UA runner (all configurations; `--llm phi3:medium` for the second backbone, `--suffix imp` for implicit variants; `--answer-prompts p0,p1,p2` for the answer-prompt ablation, default `p0` reproduces the reported runs). |
| `run_threshold_sweep.py` | Similarity-threshold sweep (0.85, 0.90, 0.95). |
| `run_a3_noretire.py`, `lww_memory.py`, `noretire_memory.py`, `run_tsread_arm.py` | Component-isolation and baseline arms (SS-noRetire, last-write-wins, TS-Read). |
| `run_mem0_arm.py`, `run_mem0_arm_patched.py` | External-system arm (Mem0). The patched runner contains the 4-line normalization patch for the mem0 2.0.17 string-fact crash and leaves the library otherwise unchanged. |
| `dp2_make_pairs.py`, `dp2_embed.py` | Similarity analysis pipeline behind the update-utterance scatter plot (Fig. 8). |
| `chain_a.sh` | End-to-end replication chain. |
| `results/canonical/` | Canonical run outputs (Batch A, Batch B, Phi-3 cross-backbone). Two Batch A snapshots are included; the analysis pipeline reads `v4_holdout_20260602_215257.json`. |
| `results/` | Per-event traces for every arm (`behavior_*.json`, including stored context and generated answer per scenario), threshold sweeps, similarity data. `behavior_LWWMemory_fn.json` is the LWW arm on the F/N subset (`run_behavior_probe.py --scenario-dir scenarios/fn --models LWWMemory --all-types --suffix fn`). |
| `figures/data/` | Per-figure data files for all data figures in the manuscript. |
| `analysis/` | Bootstrap confidence intervals, the embedder-replacement analysis, and the prompt-ablation aggregation (`prompt_ablation_aggregate.py`) with its diagnostic scorer (`prompt_ablation_diagnostic.py`). |
| `run_altemb.py`, `calib_quantile.py`, `make_engine_alt.py`, `engine/epsem_supersession_alt.py` | Detector transfer to alternative embedders (bge-small-en-v1.5, gte-small): quantile mapping of the five thresholds on the Forgetting and Noise scenarios, and a full replay under unchanged or mapped thresholds. `epsem_supersession_alt.py` is a derived copy of the frozen engine in which the five threshold literals are exposed as attributes; `make_engine_alt.py` regenerates it from the frozen file and reports every substitution. |
| `probe_llm_calls.py` | Counts language-model and embedding calls and prompt/response sizes per configuration on one scenario (latency attribution). |
| `run_prompt_ablation.sh`, `run_embedder_transfer.sh`, `run_fn_canonical.sh` | Chains for the second-round experiments (see below). |

## Environment

The benchmark runs on any recent Ollama installation. Scoring is deterministic given the generated outputs, so UA measurements on other stacks are valid in their own right.

Exact reproduction of the numbers reported in the paper additionally requires the pinned stack below, since LLM serving stacks do not guarantee identical generation across versions. The runners check the Ollama version and abort on a mismatch to protect this. Generation can also vary with the host's GPU numeric kernels (OS build), so on a different host, similarity judgments that sit near the detector threshold may occasionally flip; scoring itself is deterministic given the generated outputs.

- Ollama **0.24.0** (pinned for exact reproduction)
- Models: `qwen2.5:14b-instruct-q4_K_M` (primary backbone), `phi3:medium` (cross-backbone check), `nomic-embed-text` (detector embedding space)
- Python 3.11+ with `requests`, `pyyaml`, and `numpy` (see `requirements.txt`)
- Mem0 arm only: `pip install mem0ai==2.0.17`
- `analysis/embedder_ablation.py`, `run_altemb.py`, `calib_quantile.py` only: `pip install sentence-transformers` (bge-small-en-v1.5 and gte-small are downloaded from the Hugging Face hub on first use)

## Reproducing the reported numbers

```bash
# Verify engine integrity
shasum -a 256 -c MANIFEST.sha256

# Canonical runs (recall, CCR, store-level UA)
python3 run_holdout_v4.py
python3 run_holdout_v5.py

# Behavior-level UA, sweeps, baselines, implicit variants, Mem0
./chain_a.sh

# Similarity analysis (Fig. 8)
python3 dp2_make_pairs.py
python3 dp2_embed.py
```

On the verified host stack, a full repetition of the Update Conflict scenarios reproduces every event outcome, every context length, and every generated answer identically.

Runner scripts were rewritten from the archived execution tree to repository-relative paths; the engine files are unchanged in every executable statement (comment-level cleanup only). `dp2_make_pairs.py` regenerates `dp2_pairs.json` from the released scenarios and canonical results with an exact match to the shipped copy.

## Second-round experiments

Two experiments were added during peer review. Their runners and per-scenario outputs are included so that both can be replayed on the pinned stack. The `runner_sha256` fields in these outputs refer to the execution copies of the scripts, which were rewritten to repository-relative paths for release.

**Answer-prompt ablation.** `run_prompt_ablation.sh` replays the six behavior-level configurations on the 24 Update Conflict scenarios once each and scores three answer prompts from the same replay (`p0` verbatim listing, `p1` own words, `p2` no form instruction). Outputs: `results/behavior_*_prompt_ablation.json` (per scenario: stored context, answers by prompt, behavior-level outcome by prompt). `analysis/prompt_ablation_aggregate.py` reproduces the reported table; `analysis/prompt_ablation_diagnostic.py` applies the tolerant diagnostic scorer used to attribute the decline. `results/behavior_RagMemory_U003_replay{1,2}.json` are two further replays of one RAG scenario whose retrieved context differed in one distractor line from the reported run.

**Detector transfer to alternative embedders.** `calib_quantile.py` maps the five detector thresholds to the same quantiles of the similarity distribution of bge-small-en-v1.5 and gte-small, estimated on the Forgetting and Noise scenarios (`results/calib_map.json`). `run_embedder_transfer.sh` then replays the full configuration under each embedder with the thresholds unchanged and with the mapped thresholds (`results/altemb_{bge,gte}_{fixed,quantile}_{uc,fn}.json`). `run_fn_canonical.sh` replays the reported configuration on the Forgetting and Noise scenarios with event logging (`results/behavior_EpSemSupersession_fn_canonical.json`), which gives the archive-operation count of the operating point.

## Licenses

- Code (engine, runners, generator, analysis): [MIT](LICENSE)
- Scenario data and result data (`scenarios/`, `results/`, `figures/data/`): [CC BY 4.0](LICENSE-DATA)

## Citation

The companion manuscript is under review at IEEE Access. A citation entry will be added upon publication. Until then, please cite this repository.

## Contact

Hoyeon Lee, Shinhan University — hoyeon@shinhan.ac.kr
