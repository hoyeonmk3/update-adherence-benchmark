#!/bin/sh
# Prompt ablation: 6 configurations x 24 Update Conflict scenarios x {p0,p1,p2} in one replay per configuration.
# Stack: ollama 0.24.0 (version guard inside the runner).
cd "$(dirname "$0")" || exit 1
rm -f A_DONE.marker
echo "=== stage A start $(date) ===" > prompt_ablation.log
curl -s http://localhost:11434/api/version >> prompt_ablation.log; echo >> prompt_ablation.log
sw_vers -buildVersion >> prompt_ablation.log
python3 run_behavior_probe.py --answer-prompts p0,p1,p2 --suffix prompt_ablation >> prompt_ablation.log 2>&1
echo "=== stage A end $(date) ===" >> prompt_ablation.log
echo done > A_DONE.marker
