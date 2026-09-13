#!/bin/sh
cd "$(dirname "$0")" || exit 1
rm -f D_DONE.marker
echo "=== stage D start $(date) ===" > embedder_transfer.log
curl -s http://localhost:11434/api/version >> embedder_transfer.log; echo >> embedder_transfer.log; sw_vers -buildVersion >> embedder_transfer.log
for arm in "bge quantile" "gte quantile" "bge fixed" "gte fixed"; do
  set -- $arm
  echo "=== arm $1 $2 start $(date) ===" >> embedder_transfer.log
  python3 run_altemb.py --embedder $1 --calib $2 --sets uc,fn >> embedder_transfer.log 2>&1
  echo "=== arm $1 $2 end $(date) ===" >> embedder_transfer.log
  echo done > D_$1_$2.marker
done
echo "=== stage D end $(date) ===" >> embedder_transfer.log
echo done > D_DONE.marker
