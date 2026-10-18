"""Launch classification and regression benchmark runs."""
from __future__ import annotations
import argparse, json, os, shlex, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GROUPS=['tabular','deep_strings','pretrained_lm','graphs']

def commands(experiment,group,seed,dataset,output,python=sys.executable):
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if experiment=='classification':
        code=ROOT/'classification/code';out=output/'runs'/group/f'seed_{seed}'
        common=['--dataset-root',str(dataset),'--output-dir',str(out),'--split-methods','homology_30','--activities','all','--folds','single_split','--seed',str(seed),'--prediction-root',str(output/'peptide_length_analysis/prediction_shards')]
        script=code/('peptide_benchmark_'+group+'.py')
        if group=='tabular':variants=[['--models','all_tabular','--n-jobs','32']]
        elif group=='deep_strings':
            variants=[]
            for models,epochs,patience in [('cnn_fasta,cnn_smiles',50,8),('bilstm_fasta,bilstm_smiles',70,10),('cnn_bilstm_fasta,cnn_bilstm_smiles',70,10),('transformer_fasta,transformer_smiles',80,10)]:
                variants.append(['--models',models,'--epochs',str(epochs),'--patience',str(patience),'--min-delta','0.001','--batch-size','128','--emb-dim','96','--hidden-dim','96','--device','cuda'])
        elif group=='pretrained_lm':
            variants=[['--models',models,'--epochs','30','--patience','5','--min-delta','0.001','--batch-size','128','--embed-batch-size','8','--hidden-dim','128','--device','cuda'] for models in ['esm2_mlp,protbert_mlp,dplm_mlp,chemberta_mlp,peptideclm_mlp']]
        else:
            variants=[['--models',models,'--epochs',str(epochs),'--patience','15','--min-delta','0.001','--batch-size','64','--hidden-dim','96','--layers','3','--device','cuda'] for models,epochs in [('gcn_residue_23d,gat_residue_23d,gin_residue_23d',120),('gcn_atom_edge_aware,gat_atom_edge_aware,gine_atom_edge_aware',150)]]
    else:
        code=ROOT/('regression/base/code' if experiment=='regression' else 'regression/half_life_smiles/code')
        script=code/('regression_'+group+'.py')
        common=['--dataset-root',str(dataset),'--output-dir',str(output/'results'),'--checkpoint-root',str(output/'checkpoints'),'--cache-dir',str(output/'cache'/group),'--seed',str(seed),'--min-free-gb','50']
        extras={'tabular':[],'deep_strings':['--batch-size','64','--device','cuda'],'pretrained_lm':['--batch-size','16','--embed-batch-size','8','--device','cuda'],'graphs':['--batch-size','64','--device','cuda']}[group]
        if experiment=='half-life-smiles':
            models={'tabular':'rf_ecfp4,xgboost_ecfp4,lightgbm_ecfp4','deep_strings':'cnn_smiles,bilstm_smiles,cnn_lstm_smiles,transformer_smiles','pretrained_lm':'chemberta_mlp,peptideclm_mlp','graphs':'gcn_atom_edge_aware,gat_atom_edge_aware,gine_atom_edge_aware'}[group]
            extras=[*extras,'--tasks','half_life_len50','--models',models]
        variants=[extras]
    return [(code,[python,str(script),*common,*extra]) for extra in variants]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['classification','regression','half-life-smiles'])
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--group',choices=['all',*GROUPS],default='all');p.add_argument('--seeds',default='0,1,2,3,4')
    p.add_argument('--gpu',help='CUDA device index for this worker.')
    p.add_argument('--python',default=sys.executable);p.add_argument('--execute',action='store_true')
    a=p.parse_args();seeds=[int(x) for x in a.seeds.split(',')]
    if len(seeds)!=len(set(seeds)) or not set(seeds)<=set(range(5)):p.error('Seeds must be unique members of 0,1,2,3,4')
    if not seeds:p.error('Empty seeds')
    if a.execute:
        if not a.dataset_root.is_dir():p.error('Dataset root not found')
        data=a.dataset_root.resolve();out=a.output_root.resolve()
        if data==out or data in out.parents or out in data.parents:p.error('Input and output roots must not overlap')
        if ROOT==out or out in ROOT.parents:p.error('Output must not contain the release source directory')
        protected=[ROOT/'classification/code',ROOT/'regression/base/code',ROOT/'regression/half_life_smiles/code']
        if any(out==src or src in out.parents for src in protected):p.error('Output must not be inside frozen source code directories')
        a.output_root.mkdir(parents=True,exist_ok=True)
    groups=GROUPS if a.group=='all' else [a.group]
    for group in groups:
        for seed in seeds:
            gpu='' if group=='tabular' else a.gpu
            if gpu is None:gpu={'deep_strings':'0','pretrained_lm':'1','graphs':('2' if seed%2==0 else '3') if a.experiment=='classification' else '2'}[group]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
            if group=='tabular' and a.experiment!='classification':env.update(OMP_NUM_THREADS='32',OPENBLAS_NUM_THREADS='32',MKL_NUM_THREADS='32')
            for cwd,cmd in commands(a.experiment,group,seed,a.dataset_root,a.output_root,a.python):
                print(json.dumps({'cwd':str(cwd),'CUDA_VISIBLE_DEVICES':gpu,'argv':cmd,'execute':a.execute}),flush=True)
                if a.execute:
                    if shutil.disk_usage(a.output_root).free<50*1024**3:raise SystemExit('PAUSED_LOW_DISK: less than 50 GiB')
                    subprocess.run(cmd,cwd=cwd,env=env,check=True)

if __name__=='__main__':main()
