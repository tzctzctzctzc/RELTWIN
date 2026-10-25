#!/usr/bin/env python3
"""Validate released adaptation files without loading pickle or model tensors."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f"Missing/incomplete checkpoint: {path}. Run git lfs pull.")
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {path}')
    return {'release': manifest['release'], 'files_verified': len(manifest['files']),
            'tensor_or_pickle_deserialization': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
