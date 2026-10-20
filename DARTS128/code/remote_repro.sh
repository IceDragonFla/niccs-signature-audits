#!/usr/bin/env bash
# remote_repro.sh — run from the code/ directory on the LOCAL side.
# Syncs patched DARTS sources + attack tools to the remote WSL and runs the
# full 2M-signature attack on an INDEPENDENT (seeded) key with W=16 workers.
set -e
R="${REMOTE:-user@remote-host}"
P=2222
ROOT="${NICCS_ROOT:-/path/to/NICCS_workspace}"
DARTS_ROOT="$ROOT/01_公钥密码_数字签名/DARTS/Implementations/Reference_Implementation"
KEY_SEED="${KEY_SEED:-7f3a9c1e5b2d84f06a19c3e75b2d84f06a19c3e75b2d84f06a19c3e75b2d84f0}"
NS="${NS:-2000000}"
W="${W:-16}"

echo "== sync patched DARTS sources (128/256/512) =="
ssh -p $P $R "mkdir -p ~/darts_work/attack ~/darts_work/repro && rm -f ~/darts_work/repro/pk*.bin ~/darts_work/repro/sk*.bin ~/darts_work/repro/truth*.bin ~/darts_work/repro/pub*.bin ~/darts_work/repro/sig*.bin ~/darts_work/repro/labels*.bin"
for d in DARTS128 DARTS256 DARTS512; do
  tar czf - -C "$DARTS_ROOT" $d | ssh -p $P $R "tar xzf - -C ~/darts_work && sed -i 's/__int64_t/int64_t/g' ~/darts_work/$d/fft.c"
done
echo "== sync attack tools into a fresh repro dir =="
tar czf - attack.py forge.c gen_dump.c label_extract.c sign_dump.c run_parallel.sh diag.py | \
  ssh -p $P $R "tar xzf - -C ~/darts_work/repro"

echo "== launch remote pipeline (key seed ${KEY_SEED:0:8}..., NS=$NS, W=$W) =="
ssh -p $P $R "cd ~/darts_work/repro && DARTS_KEY_SEED=$KEY_SEED NS=$NS W=$W bash run_parallel.sh"
