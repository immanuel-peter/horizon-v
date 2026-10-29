"""Apply and audit an exact source patch without overwriting unexpected edits."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def apply(root, recipe_path, patch_path):
    recipe=json.loads(recipe_path.read_text())
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=recipe['revision']:raise RuntimeError(f'Source revision mismatch: {head}')
    patch=patch_path.resolve()
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=recipe['patch_sha256']:raise RuntimeError('Patch digest mismatch')
    mismatches=[name for name,expected in recipe['files'].items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected]
    if mismatches:
        for name in recipe['files']:
            original=subprocess.check_output(['git','-C',str(root),'show',f'HEAD:{name}'])
            if original!=(root/name).read_bytes():raise RuntimeError(f'Preserve unexpected or partially applied edits before patching: {name}')
        subprocess.run(['git','apply','--check',str(patch)],cwd=root,check=True)
        subprocess.run(['git','apply',str(patch)],cwd=root,check=True)
    for name,expected in recipe['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Source audit mismatch: {name}')
    print(json.dumps({'revision':head,'verified_source_files':len(recipe['files']),'patch_sha256':recipe['patch_sha256']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('recipe',type=Path);p.add_argument('patch',type=Path);a=p.parse_args();apply(a.root,a.recipe,a.patch)
