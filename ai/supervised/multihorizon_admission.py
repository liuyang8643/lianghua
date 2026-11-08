"""Shared semantic factor admission and actual sealed input verification."""
import json,re
from pathlib import Path
from factor_db import FORBIDDEN_FACTOR_NAMES
from utils.atomic_file import file_sha256

REFERENCE_KEYS={'name','alias_of','numeric_alias_of','source_name','canonical_name','original_name','factor_name'}
def references(record):
    result=[]
    if isinstance(record,dict):
        for key,value in record.items():
            if key in REFERENCE_KEYS and isinstance(value,str):result.append(value)
            elif isinstance(value,(dict,list)):result.extend(references(value))
    elif isinstance(record,list):
        for value in record:result.extend(references(value))
    return result

def rejected_factors(records):
    records=list(records);blocked={n.casefold() for n in FORBIDDEN_FACTOR_NAMES};reasons={}
    def forbidden(name):return name.casefold() in blocked or any(p.casefold() in blocked for p in re.split(r'[^a-zA-Z0-9]+',name))
    changed=True
    while changed:
        changed=False
        hashes={r.get('metadata',{}).get('implementation_hash') for r in records if r['name'].casefold() in blocked}-{None}
        values={r.get('scores_sha256') for r in records if r['name'].casefold() in blocked}-{None}
        for r in records:
            reason=None
            if any(forbidden(n) for n in references(r)):reason='forbidden_name_or_alias'
            elif r.get('metadata',{}).get('implementation_hash') in hashes:reason='same_forbidden_implementation'
            elif r.get('scores_sha256') in values:reason='same_forbidden_numeric_scores'
            if reason:
                reasons[r['name']]=reason
                if r['name'].casefold() not in blocked:blocked.add(r['name'].casefold());changed=True
    return reasons

def require_catalog(root):
    binding=root/'selection/binding_audit.json'
    if binding.exists():
        for name,sha in json.loads(binding.read_text())['artifacts'].items():
            if file_sha256(root/'selection'/name)!=sha:raise ValueError('candidate catalogue binding changed: '+name)
    records=json.loads((root/'selection/selected_3d.json').read_text(encoding='utf8'))['selected']
    rejected=rejected_factors(records)
    if rejected:raise ValueError('forbidden factor aliases: '+str(rejected))

def verify_data(root):
    directory=root/'features';manifest=json.loads((directory/'data_manifest.json').read_text())
    for name,sha in manifest.items():
        if file_sha256(directory/name)!=sha:raise ValueError('sealed data changed: '+name)
    protocol=root/'protocol.json'
    if protocol.exists():
        c=json.loads(protocol.read_text(encoding='utf8'))
        if c.get('verified_feature_reuse_root'):
            import numpy as np
            original=Path(c['verified_feature_reuse_root'])/'features/codes.npy'
            np.testing.assert_array_equal(np.load(directory/'codes.npy'),np.load(original))
    require_catalog(root)
    return file_sha256(directory/'data_manifest.json')
