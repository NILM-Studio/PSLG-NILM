"""Independent frozen-teacher and activity-clustered conditional diagnostics."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from .mechanism_data import runs


def features(y):
    y = np.asarray(y, float)
    mean = uniform_filter1d(y, 5, mode='nearest')
    variance = np.maximum(uniform_filter1d(y*y, 5, mode='nearest') - mean*mean, 0)
    return np.stack((y, mean, np.sqrt(variance), np.gradient(y) if len(y)>1 else y*0), 1)


def fit_teacher(records, seed=717):
    rng = np.random.default_rng(seed)
    xx, zz = [], []
    for r in records:
        ix = rng.choice(len(r['y']), min(len(r['y']), 256), replace=False)
        xx.append(features(r['y'])[ix]); zz.append(r['z'][ix])
    x, z = np.concatenate(xx), np.concatenate(zz)
    ix = rng.choice(len(z), min(len(z), 100000), replace=False)
    teacher = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5,
                                     class_weight='balanced', random_state=seed, n_jobs=4)
    teacher.fit(x[ix], z[ix])
    return teacher


def boundary_score(z, pred, tolerance=2):
    """One-to-one boundary matching; unmatched boundaries remain errors."""
    a, b = np.flatnonzero(np.diff(z)) + 1, np.flatnonzero(np.diff(pred)) + 1
    matches, used_a, used_b = [], set(), set()
    candidates = sorted((abs(int(x)-int(y)), i, j) for i,x in enumerate(a) for j,y in enumerate(b)
                        if abs(x-y) <= tolerance)
    for distance, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); matches.append(distance * 6)
    return dict(true=len(a), predicted=len(b), matched=len(matches), error_seconds=matches)


def teacher_diagnostics(records, generated, teacher, scale, seed=91, n_states=5):
    if len(records) != len(generated):
        raise ValueError('Unpaired activity lists')
    classes = range(1, n_states + 1)
    cm = np.zeros((n_states, n_states), np.int64); rows = []; bystate = {k: [[], []] for k in classes}
    boundaries = dict(true=0, predicted=0, matched=0, error_seconds=[])
    groups = {}; duration_groups = {}; state_durations = {k:[[],[]] for k in classes}
    state_rates = {k:[[],[]] for k in classes}
    boundary_pairs = {}; all_real=[]; all_generated=[]; energies=[[],[]]; peaks=[[],[]]
    for r, y in zip(records, generated):
        ref, z = r['y'], r['z']
        if len(y) != len(z) or not np.isfinite(y).all():
            raise ValueError('Invalid generated activity')
        pred = teacher.predict(features(y))
        all_real.append(ref); all_generated.append(y)
        energies[0].append(y.sum()*6/3600000);energies[1].append(ref.sum()*6/3600000)
        peaks[0].append(y.max());peaks[1].append(ref.max())
        cm += confusion_matrix(z, pred, labels=list(classes))
        b = boundary_score(z, pred)
        for k in boundaries:
            boundaries[k] += b[k]
        f1 = f1_score(z, pred, labels=list(classes), average='macro', zero_division=0)
        groups.setdefault(r['id'], []).append(f1)
        rows.append(dict(activity=r['id'], macro_f1=float(f1),
                         energy_error_kwh=float((y.sum()-ref.sum())*6/3600000),
                         negative_fraction=float((y<0).mean())))
        for k in bystate:
            bystate[k][0].extend(y[z == k]); bystate[k][1].extend(ref[z == k])
            inside=(z[1:]==k)&(z[:-1]==k)
            state_rates[k][0].extend(np.diff(y)[inside]/6)
            state_rates[k][1].extend(np.diff(ref)[inside]/6)
        for labels, slot in ((pred,0),(z,1)):
            for lo,hi in runs(labels):state_durations[int(labels[lo])][slot].append((hi-lo)*6)
        for lo,hi in runs(z):
            seconds=(hi-lo)*6
            band='short' if seconds<180 else 'medium' if seconds<600 else 'long'
            key=f'{int(z[lo])}/{band}'
            entry=duration_groups.setdefault(key,dict(blocks=0,activities=set(),correct=0,points=0))
            entry['blocks']+=1;entry['activities'].add(r['id'])
            entry['correct']+=int((pred[lo:hi]==z[lo:hi]).sum());entry['points']+=int(hi-lo)
            if lo:
                key=f'{int(z[lo-1])}->{int(z[lo])}'
                pair=boundary_pairs.setdefault(key,[[],[]])
                pair[0].append(float(y[lo]-y[lo-1]));pair[1].append(float(ref[lo]-ref[lo-1]))
    state = {}
    for k, (a,b) in bystate.items():
        state[str(k)] = dict(points=len(a), power_w1=float(wasserstein_distance(a,b))) if a else None
        if a:
            ar,br=state_rates[k];ad,bd=state_durations[k]
            state[str(k)]['internal_rate_w1']=float(wasserstein_distance(ar,br)) if ar and br else None
            state[str(k)]['duration_w1_seconds']=float(wasserstein_distance(ad,bd)) if ad and bd else None
            state[str(k)]['predicted_blocks']=len(ad)
    duration_groups={k:dict(blocks=v['blocks'],activities=len(v['activities']),points=v['points'],
                            adherence=v['correct']/v['points']) for k,v in duration_groups.items()}
    boundary_pairs={k:dict(count=len(a),jump_w1=float(wasserstein_distance(a,b))) for k,(a,b) in boundary_pairs.items()}
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(values,len(values),replace=True).mean()) for _ in range(1000)]
    precision = boundaries['matched']/max(1,boundaries['predicted'])
    recall = boundaries['matched']/max(1,boundaries['true'])
    return dict(confusion_matrix=cm.tolist(), labels=list(classes),
                macro_state_f1=float(np.mean([2*cm[k,k]/max(1,cm[k].sum()+cm[:,k].sum()) for k in range(n_states)])),
                mean_activity_f1=float(values.mean()), activity_bootstrap_ci95=np.quantile(boots,[.025,.975]).tolist(),
                independent_activities=len(groups), per_state=state,
                macro_state_power_w1=float(np.mean([s['power_w1'] for s in state.values() if s])),
                state_duration_strata=duration_groups,boundary_pair_jump=boundary_pairs,
                global_power_w1=float(wasserstein_distance(np.concatenate(all_generated),np.concatenate(all_real))),
                energy_kwh_w1=float(wasserstein_distance(*energies)),peak_w1=float(wasserstein_distance(*peaks)),
                negative_fraction=float((np.concatenate(all_generated)<0).mean()),
                boundary_precision=precision,boundary_recall=recall,
                boundary_mae_seconds=float(np.mean(boundaries['error_seconds'])) if boundaries['error_seconds'] else None,
                boundary_tolerance_seconds=12, activities=rows,
                limitations=['Teacher labels are predictions, not appliance functional ground truth',
                             'Requested-state stratification must be read together with confusion matrix',
                             'Energy differences compare conditional distributions, not paired reconstruction accuracy'])


def waveform_quality(real, generated, scale):
    """Unpaired distribution quality. No classifier or state identities required."""
    from scipy.spatial.distance import cdist
    import hashlib
    def descriptors(rows):
        values=[];spectra=[];rates=[]
        grid=np.linspace(0,1/12,33)
        for row in rows:
            y=np.asarray(row,float)
            if not len(y) or not np.isfinite(y).all():raise ValueError('Invalid waveform')
            centered=y-y.mean()
            spec=np.abs(np.fft.rfft(centered))**2/max(1,len(y))
            spectra.append(np.interp(grid,np.fft.rfftfreq(len(y),d=6),np.log1p(spec)))
            rate=np.diff(y)/6
            rates.extend(rate)
            acf=float(np.mean(centered[:-1]*centered[1:])/max(np.mean(centered**2),1e-8)) if len(y)>1 else 0.
            values.append([y.mean(),y.std(),y.max(),np.mean(abs(rate)) if len(rate) else 0.,acf])
        return np.asarray(values),np.asarray(spectra),np.asarray(rates)
    r,rs,rr=descriptors(real);g,gs,gr=descriptors(generated)
    normalizer=np.array([scale,scale,scale,scale/6,1.])
    nearest_real=np.full(len(r),np.inf);nearest_generated=[]
    for start in range(0,len(g),256):
        distances=cdist(g[start:start+256]/normalizer,r/normalizer)/np.sqrt(5)
        nearest_real=np.minimum(nearest_real,distances.min(0))
        nearest_generated.extend(distances.min(1))
    return dict(global_power_w1=float(wasserstein_distance(np.concatenate(generated),np.concatenate(real))),
        energy_kwh_w1=float(wasserstein_distance([y.sum()*6/3600000 for y in generated],[y.sum()*6/3600000 for y in real])),
        peak_w1=float(wasserstein_distance(g[:,2],r[:,2])),
        rate_w1=float(wasserstein_distance(gr,rr)) if len(gr) and len(rr) else None,
        waveform_std_w1=float(wasserstein_distance(g[:,1],r[:,1])),
        acf_lag1_w1=float(wasserstein_distance(g[:,4],r[:,4])),
        mean_log_psd_l1=float(np.abs(gs.mean(0)-rs.mean(0)).mean()),
        negative_fraction=float((np.concatenate(generated)<0).mean()),
        descriptor_spread_real=(r/normalizer).std(0).tolist(),
        descriptor_spread_generated=(g/normalizer).std(0).tolist(),
        reference_to_generated_distance=float(nearest_real.mean()),
        generated_to_reference_distance=float(np.mean(nearest_generated)),
        unique_waveform_fraction=len({hashlib.sha256(np.asarray(y,dtype=np.float32).tobytes()).hexdigest() for y in generated})/len(generated),
        reference_waveforms=len(real),generated_waveforms=len(generated),
        descriptor_order=['mean_power','power_std','peak','mean_absolute_rate','acf_lag1'])


def evaluate(records, generated, teacher=None, scale=1500., seed=91, n_states=5, conditional=True):
    if not records or len(records)!=len(generated):raise ValueError('Unpaired activity lists')
    if any(len(r['y'])!=len(y) for r,y in zip(records,generated)):raise ValueError('Length mismatch')
    result=dict(schema='waveform_quality_v2',
        waveform_quality=waveform_quality([r['y'] for r in records],generated,scale),
        independent_activities=len({r['id'] for r in records}),
        conditional_quality=None,auxiliary_classifier=None,
        limitations=['Duration is supplied to the generator; duration planning accuracy is not measured',
                     'Rate and spectrum can reflect sensor noise; robustness requires known clean references',
                     'Descriptor spread and nearest-neighbor coverage are diagnostics, not calibrated precision/recall',
                     'No paired pointwise reconstruction loss is used to rank free waveform samples'])
    if conditional:
        bystate={str(k):[[],[]] for k in range(1,n_states+1)};pairs={};strata={}
        for record,y in zip(records,generated):
            z,real=record['z'],record['y']
            for lo,hi in runs(z):
                key=str(int(z[lo]))
                bystate[key][0].append(real[lo:hi]);bystate[key][1].append(y[lo:hi])
                duration=(hi-lo)*6
                band='short' if duration<180 else 'medium' if duration<600 else 'long'
                bins=strata.setdefault(key+'/'+band,[[],[]]);bins[0].append(real[lo:hi]);bins[1].append(y[lo:hi])
                if lo:
                    pair=pairs.setdefault(f'{int(z[lo-1])}->{int(z[lo])}',[[],[]])
                    pair[0].append(float(real[lo]-real[lo-1]));pair[1].append(float(y[lo]-y[lo-1]))
        per_state={k:waveform_quality(a,b,scale) if a else None for k,(a,b) in bystate.items()}
        result['conditional_quality']=dict(grouping='requested labels vs existing reference pseudo-labels; no reclassification',
            per_state=per_state,
            macro_state_power_w1=float(np.mean([v['global_power_w1'] for v in per_state.values() if v])),
            state_duration_strata={k:waveform_quality(a,b,scale) for k,(a,b) in strata.items()},
            boundary_pair_jump={k:dict(count=len(a),jump_w1=float(wasserstein_distance(a,b))) for k,(a,b) in pairs.items()})
        if teacher is not None:
            result['auxiliary_classifier']=teacher_diagnostics(records,generated,teacher,scale,seed,n_states)
    return result
