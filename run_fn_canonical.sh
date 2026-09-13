#!/bin/sh
# Stage E: canonical EpSemSupersession (frozen engine 160a84ab, nomic, 0.92) on F/N-48 with event logging.
# Purpose: F/N archive-count comparator for Exp D (canonical value was never recorded). Runs only after stage D finishes.
cd "$(dirname "$0")" || exit 1
rm -f E_DONE.marker
echo "=== stage E queued $(date) — waiting for D_DONE.marker ===" > fn_canonical.log
while [ ! -f D_DONE.marker ]; do sleep 120; done
echo "=== stage E start $(date) ===" >> fn_canonical.log
curl -s http://localhost:11434/api/version >> fn_canonical.log; echo >> fn_canonical.log; sw_vers -buildVersion >> fn_canonical.log
shasum -a 256 engine/epsem_supersession.py run_behavior_probe.py >> fn_canonical.log
python3 run_behavior_probe.py --models EpSemSupersession --scenario-dir scenarios/fn --all-types --no-answers --suffix fn_canonical >> fn_canonical.log 2>&1
echo "=== stage E end $(date) ===" >> fn_canonical.log
echo done > E_DONE.marker
