#!/usr/bin/env bash
# after.sh PID [PID...] -- command...   : wait for every PID to exit, then run the command.
# Chains a card's jobs without a scheduler: the next job starts when the card frees.
pids=()
while [ "$1" != "--" ] && [ $# -gt 0 ]; do pids+=("$1"); shift; done
shift
for p in "${pids[@]}"; do while kill -0 "$p" 2>/dev/null; do sleep 15; done; done
exec "$@"
