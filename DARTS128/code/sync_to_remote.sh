#!/usr/bin/env bash
# sync_to_remote.sh — push the 1-2b patched DARTS sources + attack workspace
# to the remote compute server (${REMOTE:-user@remote-host}:2222) once it is reachable.
# Run from the DARTS128_12b_attack/code directory.
set -e
R="${REMOTE:-user@remote-host}"
P=2222
ROOT="${NICCS_ROOT:-/path/to/NICCS_workspace}"

ssh -p $P $R "mkdir -p ~/darts_work/attack"
for d in DARTS128 DARTS256 DARTS512; do
  scp -q -P $P -r "$ROOT/01_公钥密码_数字签名/DARTS/Implementations/Reference_Implementation/$d" \
      $R:~/darts_work/
  ssh -p $P $R "sed -i 's/__int64_t/int64_t/g' ~/darts_work/$d/fft.c && \
                (cd ~/darts_work/$d && make test 2>&1 | tail -1)"
done
scp -q -P $P sign_dump.c gen_dump.c label_extract.c forge.c attack.py run_all.sh \
    check256512.sh $R:~/darts_work/attack/
echo "sync done; artifacts on remote: ~/darts_work/attack/"
