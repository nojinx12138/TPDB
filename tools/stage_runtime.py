"""Prepare a benchmark runtime workspace."""
import argparse, hashlib, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',choices=['classification','regression','half-life-smiles']);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
 source=ROOT/{'classification':'classification/code','regression':'regression/base/code','half-life-smiles':'regression/half_life_smiles/code'}[a.experiment]
 target=a.output_root.resolve()
 if target==ROOT or target in ROOT.parents:raise ValueError('Runtime workspace cannot contain release sources')
 protected=[ROOT/'classification/code',ROOT/'regression/base/code',ROOT/'regression/half_life_smiles/code']
 if any(target==p or p in target.parents for p in protected):raise ValueError('Runtime cannot be inside frozen source directories')
 code=target if a.experiment=='classification' else target/'code'
 for file in source.rglob("*"):
  if not file.is_file():continue
  dst=code/file.relative_to(source)
  if dst.resolve()==file.resolve():raise ValueError('Runtime must be separate from source')
  if dst.exists() and dst.read_bytes()!=file.read_bytes():raise ValueError('Existing runtime code differs: '+str(dst))
  dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copy2(file,dst)
  assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(file.read_bytes()).digest()
 print('Runtime code directory:',code)
if __name__=='__main__':main()
