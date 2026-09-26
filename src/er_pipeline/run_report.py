"""Summarize a completed, audited run without loading the large pair tables."""
import importlib.metadata
import json
import platform
from pathlib import Path
from .common import dump_json


def environment():
    result = {'python': platform.python_version(), 'packages': {}}
    for name in ('numpy', 'rapidfuzz', 'pyarrow', 'lightgbm', 'torch', 'sentence-transformers',
                 'transformers', 'peft', 'faiss-cpu', 'lightning-sdk', 'boto3'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    return result


def build(work, sample):
    work = Path(work)
    def read(path):
        return json.loads((work / path).read_text())
    result = dict(sample_only=sample, environment=environment(),
                  dataset=read('cleaned/cleaning_report.json'),
                  coverage=read('test/submission_coverage.json'),
                  candidate_recall=read('train/candidate_recall.json'),
                  validation=read('train/validation_f05.json'),
                  model=read('model/model_metadata.json'),
                  encoder=read('encoder/complete.json'),
                  validation_note='Threshold/checkpoint selected on validation; not an unbiased test or leaderboard score.')
    destination = work / ('sample_output' if sample else 'output')
    dump_json(destination / 'run_report.json', result)
    return result
