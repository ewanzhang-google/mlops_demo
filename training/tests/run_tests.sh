#!/usr/bin/env bash
pip install -r training/requirements.txt
python -m pytest --cov=training --cov-config=training/tests/coverage.conf --cov-report=term-missing --cov-fail-under=60 -s
rtrn_code=$?
coverage erase
exit $rtrn_code