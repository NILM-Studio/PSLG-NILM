"""Run the official COSCIGAN trainer unchanged on timestamp-aligned NILM channels."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
VENDOR=ROOT/'models/downstream_generation_models/vendor/COSCI-GAN'

def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',choices=['lstm','mlp'],default='lstm')
    a=parser.parse_args()
    execution=Path(os.environ['PSLG_EXECUTION_DIR']).resolve()
    if not execution.is_relative_to(ROOT/'runs') or not (execution/'run.json').exists():
        raise ValueError('Use scripts/create_execution.py and scripts/run_execution.py')
    torch.set_num_threads(2);np.random.seed(17);torch.manual_seed(17)
    channels=['Air_condition','Microwave','WashMachine']; frames=[];sources=[]
    for name in channels:
        path=ROOT/'datasets/self_collected'/(name+'.csv')
        df=pd.read_csv(path,nrows=86400,usecols=['datetime','active power'])
        df['datetime']=pd.to_datetime(df['datetime']);df=df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        series=df['active power'].resample('6s',origin='epoch').mean().rename(name)
        frames.append(series)
        sources.append({'path':str(path),'size_bytes':path.stat().st_size,'rows_read':len(df),
            'prefix_1m_sha256':hashlib.sha256(path.open('rb').read(1<<20)).hexdigest()})
    aligned=pd.concat(frames,axis=1,join='inner').dropna()
    assert len(aligned)>=16*64
    assert np.all(np.diff(aligned.index.asi8)==6_000_000_000), 'Non-contiguous common grid'
    starts=np.linspace(0,len(aligned)-64,16,dtype=int)
    raw=np.stack([aligned.iloc[i:i+64].to_numpy(np.float32) for i in starts])
    low=raw.min(axis=(0,1),keepdims=True);high=raw.max(axis=(0,1),keepdims=True)
    assert np.all(high>low)
    normalized=(raw-low)/(high-low)
    work=execution/'analysis/upstream/Code';work.mkdir(parents=True,exist_ok=False)
    dataset=work.parent/'Dataset';dataset.mkdir()
    # Official CSV is channel-major: [channel0 time..., channel1 time..., ...].
    csv=dataset/'self_collected_aligned.csv'
    pd.DataFrame(normalized.transpose(0,2,1).reshape(16,-1)).to_csv(csv,index=False)
    np.savez_compressed(execution/'analysis/real_windows.npz',raw_watts=raw,x=normalized,low=low,high=high,channels=np.array(channels),starts=starts)
    save(execution/'inputs.json',{'sources':sources,'alignment':'inner join actual datetime on common 6-second grid',
        'common_first':str(aligned.index[0]),'common_last':str(aligned.index[-1]),'common_rows':len(aligned),
        'shape':list(normalized.shape),'window_start_indices':starts.tolist(),'scaling':'per-channel min/max on smoke training windows; no held-out quality evaluation',
        'csv_sha256':hashlib.sha256(csv.read_bytes()).hexdigest()})
    kwargs=dict(n_groups=3,id=0,expId=0,dataset='self_collected_aligned',num_epochs=1,batch_size=8,n_samples=64,
        real_data_fraction=1.0,criterion='BCE',with_CD=True,LSTM_G=a.architecture=='lstm',LSTM_D=a.architecture=='lstm',
        CD_type='MLP',generator_lr=.001,discriminator_lr=.001,central_discriminator_lr=.0001,gamma_value=5.,noise_len=32)
    save(execution/'config.resolved.json',{'official_trainer_kwargs':kwargs,'data_seed':17,'official_torch_seed':0,'device':'cpu'})
    env=json.loads((execution/'environment.json').read_text())
    env.update(runtime_python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),vendor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip(),
        pip_freeze=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines())
    save(execution/'environment.json',env)
    spec=importlib.util.spec_from_file_location('cosci_official',VENDOR/'Code/Main_modules.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observations=[]
    Adam=torch.optim.Adam
    class CheckedAdam(Adam):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.record={'steps':0,'finite_gradients':True,'parameter_changed':False};observations.append(self.record)
        def step(self,*args,**kwargs):
            params=[p for group in self.param_groups for p in group['params'] if p.grad is not None]
            assert params and all(torch.isfinite(p.grad).all() for p in params)
            old=[p.detach().clone() for p in params]
            out=super().step(*args,**kwargs)
            self.record['steps']+=1
            self.record['parameter_changed']|=any(not torch.equal(p,o) for p,o in zip(params,old))
            return out
    previous=Path.cwd();start=time.monotonic()
    try:
        os.chdir(work)
        with patch.object(torch.optim,'Adam',CheckedAdam):module.COSCIGAN(**kwargs)
    finally:os.chdir(previous)
    assert len(observations)==7 and all(o['steps']==2 and o['parameter_changed'] for o in observations)
    weights=sorted((work.parent/'Results/Models/with_CD').glob('*_Generator_*.pt'))
    assert len(weights)==3
    torch.manual_seed(29);noise=torch.randn(4,32);generated=[]
    for weight in weights:
        model=module.LSTMGenerator(32,64) if a.architecture=='lstm' else module.Generator(32,64,.1)
        state=torch.load(weight,map_location='cpu',weights_only=True)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);model.eval()
        with torch.no_grad():generated.append(model(noise).numpy())
    samples=np.stack(generated,axis=-1);assert samples.shape==(4,64,3) and np.isfinite(samples).all()
    np.savez_compressed(execution/'generations/cosci_samples.npz',normalized=samples,watts=samples*(high-low)+low)
    shutil.move(str(work.parent/'Results/Models'),str(execution/'checkpoints/upstream_models'))
    result={'status':'pass','architecture':a.architecture,'official_training_function':'Main_modules.COSCIGAN',
        'input_shape':list(normalized.shape),'generated_shape':list(samples.shape),'generated_finite':True,
        'optimizer_order':['D0','D1','D2','G0','G1','G2','central_D'],'optimizer_checks':observations,
        'checkpoint_reload':'strict successful, all three generators','seconds':round(time.monotonic()-start,3),
        'limits':'one epoch interface smoke; no quality claim; central discriminator MLP only; no new primitive conditioning implemented'}
    save(execution/'analysis/cosci_smoke.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
