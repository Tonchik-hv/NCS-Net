#!/usr/bin/env bash
# Честная 5-fold CV: каждый фолд — новая модель, 300 эпох (как train.py).
# Запуск из корня репозитория NN.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${ROOT}/cross_validation/.venv/bin/python"
if [[ ! -x "$VENV" ]]; then
  echo "Создайте venv: python3 -m venv cross_validation/.venv && cross_validation/.venv/bin/pip install torch numpy scipy matplotlib"
  exit 1
fi
exec "$VENV" -u cross_validation/run_cross_validate.py --epochs 300 --resume "$@"
