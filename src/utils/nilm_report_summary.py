"""Pair seeds within households; never count random seeds as separate homes."""
import numpy as np
from nilm_experiments.nilm_lab.metrics import paired_house_ci

COMPARISONS = [('R', 'O'), ('R', 'P'), ('O', 'P'), ('B', 'P'), ('BP', 'P'), ('S', 'P'),
               ('B_full', 'P'), ('P', 'P_gru'), ('P_direct', 'P_gru'), ('P_point', 'P_gru'),
               ('H_gru', 'P_gru'), ('BP_gru', 'P_gru'), ('O_gru', 'P_gru')]


def summarize_test(results, appliance):
    rows = []
    seen = set()
    for result in results:
        trial = result['trial']
        key = (trial['model'], trial['case'], trial['seed'])
        if key in seen:
            raise ValueError('Duplicate selected trial')
        seen.add(key)
        houses = {}
        for score in result['scores']:
            n = score.get('n', 1)
            count, error = houses.setdefault((score['dataset'], score['house']), [0, 0.])
            houses[(score['dataset'], score['house'])] = [count+n, error+n*score['mae']]
        for (dataset, house), (count, error) in houses.items():
            rows.append(dict(dataset=dataset, house=house, mae=error/count,
                model=trial['model'], arm=trial['case'], seed=trial['seed'], appliance=appliance))
    summaries = []
    for model in sorted({r['model'] for r in rows}):
        current = [r for r in rows if r['model'] == model]
        cases = {r['arm'] for r in current}
        for baseline, proposed in COMPARISONS:
            if {baseline, proposed} <= cases:
                pair = [r for r in current if r['arm'] in {baseline, proposed}]
                summaries.extend(dict(s, baseline=baseline, proposed=proposed)
                    for s in paired_house_ci(pair, baseline, proposed))
    groups = {}
    for r in rows:
        groups.setdefault((r['dataset'], r['house'], r['model'], r['arm']), []).append(r)
    variation = []
    for (dataset, house, model, case), values in sorted(groups.items()):
        if len({r['seed'] for r in values}) != len(values):
            raise ValueError('Duplicate household/seed scores need record aggregation before reporting')
        x = [r['mae'] for r in values]
        variation.append(dict(dataset=dataset, house=house, model=model, case=case, seeds=len(x),
            mae_mean=float(np.mean(x)), mae_seed_std=float(np.std(x, ddof=1)) if len(x)>1 else None))
    return dict(paired_test=summaries, household_seed_variation=variation,
                detailed_metrics=detailed_metrics(results))


def detailed_metrics(results):
    """Aggregate additive errors/counts within house, then pair seeds; never average SAE."""
    records=[]
    for item in results:
        trial=item['trial']
        groups={}
        for score in item['scores']:
            groups.setdefault((score['dataset'],score['house']),[]).append(score)
        for (dataset,house),scores in groups.items():
            row=dict(dataset=dataset,house=house,model=trial['model'],case=trial['case'],seed=trial['seed'])
            for metric,denom in [('mae','n'),('active_MAE','active_n'),('inactive_MAE','inactive_n'),
                                  ('threshold_ON_MAE','threshold_ON_n')]:
                usable=[s for s in scores if s.get(metric) is not None and s.get(denom,0)>0]
                n=sum(s[denom] for s in usable)
                row[metric]=sum(s[metric]*s[denom] for s in usable)/n if n else None
                row[denom]=n
            for key in ['energy_wh','predicted_energy_wh','inactive_overprediction_wh','inactive_observed_hours',
                        'activity_tp','activity_fp','activity_fn','activity_tn']:
                row[key]=sum(s.get(key,0) for s in scores)
            energy=row['energy_wh']
            row['sae']=abs(row['predicted_energy_wh']-energy)/energy if energy else None
            tp,fp,fn,tn=[row[k] for k in ['activity_tp','activity_fp','activity_fn','activity_tn']]
            row['activity_F1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
            row['activity_recall']=tp/(tp+fn) if tp+fn else None
            row['activity_false_positive_rate']=fp/(fp+tn) if fp+tn else None
            records.append(row)
    groups={}
    for row in records:
        groups.setdefault((row['dataset'],row['house'],row['model'],row['case']),[]).append(row)
    means=[]
    metrics=['mae','active_MAE','inactive_MAE','threshold_ON_MAE','sae','activity_F1','activity_recall',
             'activity_false_positive_rate','inactive_overprediction_wh']
    for (dataset,house,model,case),values in sorted(groups.items()):
        item=dict(dataset=dataset,house=house,model=model,case=case,seeds=len(values))
        for metric in metrics:
            x=[r[metric] for r in values if r[metric] is not None]
            item[metric]=dict(mean=float(np.mean(x)) if x else None,
                              seed_std=float(np.std(x,ddof=1)) if len(x)>1 else None,n_seeds=len(x))
        means.append(item)
    pairs=[]
    by_seed={(r['dataset'],r['house'],r['model'],r['case'],r['seed']):r for r in records}
    for dataset,house,model,_ in sorted(groups):
        if any(p['dataset']==dataset and p['house']==house and p['model']==model for p in pairs):
            continue
        for control,proposed in COMPARISONS:
            left=groups.get((dataset,house,model,control),[])
            right=groups.get((dataset,house,model,proposed),[])
            if not left or not right:
                continue
            if {r['seed'] for r in left}!={r['seed'] for r in right}:
                raise ValueError('Missing paired formal seeds')
            for metric in ['mae','active_MAE','inactive_MAE']:
                changes=[]
                for a in left:
                    b=by_seed[(dataset,house,model,proposed,a['seed'])]
                    if a[metric] is not None and b[metric] is not None:
                        changes.append(dict(seed=a['seed'],delta=b[metric]-a[metric],control=a[metric],proposed=b[metric]))
                delta=[r['delta'] for r in changes]
                pairs.append(dict(dataset=dataset,house=house,model=model,control=control,proposed=proposed,
                    metric=metric,paired=changes,mean_delta=float(np.mean(delta)) if delta else None,
                    wins=sum(v<0 for v in delta),ties=sum(v==0 for v in delta),losses=sum(v>0 for v in delta)))
    return dict(per_seed=records,household_means=means,paired_differences=pairs,
                note='Delta=proposed-control; seeds describe training variation, not independent households')
