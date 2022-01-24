#!/usr/bin/env bash
set -e
pip install -r ../requirements.txt
python compile_pipeline.py -d $1