"""Export result CSVs using manuscript scheme names TR, PU, BI and MS."""
from __future__ import annotations
import argparse,csv
from pathlib import Path

SCHEMES={'traditional':'TR','Traditional':'TR','A':'TR','pu':'PU','B':'PU',
         'bdnegsamp':'BI','BDNegSamp':'BI','C':'BI','abc_mix':'MS','D':'MS','M':'MS',
         'TR':'TR','PU':'PU','BI':'BI','MS':'MS'}
SCHEME_FIELDS={'source_method','target_method','train_scheme','test_scheme','negative_scheme'}
MATRIX_FIELDS={'traditional':'TR','pu':'PU','bdnegsamp':'BI','abc_mix':'MS'}

def format_record(fields,row):
    out={MATRIX_FIELDS.get(k,k):v for k,v in row.items()}
    for field in SCHEME_FIELDS & row.keys():
        if row[field] not in SCHEMES:raise ValueError(f'Unknown scheme: {field}={row[field]!r}')
        out[field]=SCHEMES[row[field]]
    if 'combo' in row:
        if 'source_method' not in out or 'target_method' not in out:
            raise ValueError('combo requires source_method and target_method')
        out['combo']=out['source_method']+'_'+out['target_method']
    return [MATRIX_FIELDS.get(k,k) for k in fields],out

def export(source,destination):
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Use a separate output file')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open(encoding='utf-8-sig',newline='') as src,destination.open('w',encoding='utf-8',newline='') as dst:
        reader=csv.DictReader(src);fields=reader.fieldnames
        if not fields:raise ValueError('Missing CSV header: '+str(source))
        headers=[MATRIX_FIELDS.get(k,k) for k in fields]
        if len(headers)!=len(set(headers)):raise ValueError('Duplicate output headers')
        writer=csv.DictWriter(dst,fieldnames=headers);writer.writeheader()
        for row in reader:writer.writerow(format_record(fields,row)[1])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();source=a.input.resolve();output=a.output_dir.resolve()
    if source==output or (source.is_dir() and (source in output.parents or output in source.parents)):
        p.error('Input and output directories must be separate')
    files=[source] if source.is_file() else sorted(source.rglob('*.csv'))
    if not files:p.error('No CSV files found')
    for file in files:
        rel=Path(file.name) if source.is_file() else file.relative_to(source)
        export(file,output/rel)
    print(f'Exported {len(files)} CSV files to {output}')

if __name__=='__main__':main()
