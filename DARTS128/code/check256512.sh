#!/usr/bin/env bash
set -e
for d in DARTS256 DARTS512; do
  sed -i 's/__int64_t/int64_t/g' ~/darts_work/$d/fft.c
  (cd ~/darts_work/$d && make test 2>&1 | grep -ciE '\berror\b' || true) | xargs echo "$d compile-errors:"
  (cd ~/darts_work/$d && ./bin/test_DARTS 2>&1 | head -1) | xargs echo "$d roundtrip:"
done
