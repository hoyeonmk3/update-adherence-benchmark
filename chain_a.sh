#!/bin/bash
# End-to-end replication chain: A1->B1->A2->A3->C1->B2->B3 (every stage guarded by the ollama 0.24.0 pin)
cd .
PY=python3

guard() {
  curl -s http://localhost:11434/api/version | grep -q '"version":"0.24.0"' \
    || { echo "[chain] FATAL: version guard failed $(date)"; exit 2; }
}

echo "[chain] START $(date)"

guard; echo "[chain] A1 behavior layer, 6 arms — start $(date)"
$PY run_behavior_probe.py > a1.log 2>&1 || echo "[chain] A1 ERROR"
echo "[chain] A1 done $(date)"

guard; echo "[chain] B1 SS repeat run — start $(date)"
$PY run_behavior_probe.py --models EpSemSupersession --suffix rerun > b1.log 2>&1 || echo "[chain] B1 ERROR"
echo "[chain] B1 done $(date)"

guard; echo "[chain] A2 threshold sweep — start $(date)"
$PY run_threshold_sweep.py > a2.log 2>&1 || echo "[chain] A2 ERROR"
echo "[chain] A2 done $(date)"

guard; echo "[chain] A3 SS-noRetire, all 72 scenarios — start $(date)"
$PY run_a3_noretire.py > a3.log 2>&1 || echo "[chain] A3 ERROR"
echo "[chain] A3 done $(date)"

guard; echo "[chain] C1 implicit variants, 12 scenarios (SS,LWW,F) — start $(date)"
$PY run_behavior_probe.py --scenario-dir scenarios/implicit \
  --models EpSemSupersession,LWWMemory,EpSemFiltered --suffix imp > c1.log 2>&1 || echo "[chain] C1 ERROR"
echo "[chain] C1 done $(date)"

guard; echo "[chain] B2 phi3 behavior layer (SS) — start $(date)"
$PY run_behavior_probe.py --models EpSemSupersession --llm phi3:medium --suffix phi3 > b2.log 2>&1 || echo "[chain] B2 ERROR"
echo "[chain] B2 done $(date)"

guard; echo "[chain] B3 RAG variants, 2 configs (all 72, store metrics only) — start $(date)"
$PY run_behavior_probe.py --models RagMemory --rag-topk 20 --all-types --no-answers --suffix ragk20 > b3a.log 2>&1 || echo "[chain] B3a ERROR"
$PY run_behavior_probe.py --models RagMemory --query-reform --all-types --no-answers --suffix ragreform > b3b.log 2>&1 || echo "[chain] B3b ERROR"
echo "[chain] B3 done $(date)"

echo "[chain] ALL DONE $(date)"
