"""Keep activity-interval reference metrics separate from threshold metrics."""
import numpy as np
from .activity_labels import detect_activities, activity_projection
from .metrics import metrics


def activity_reference(record, cfg, dt):
    activities, _ = detect_activities(record.t, record.y, cfg, record.id, dt)
    labels, _, valid = activity_projection(len(record.t), activities, np.isfinite(record.y))
    return dict(activities=activities, labels=labels, valid=valid)


def score_record(record, power, activity_probability, reference, threshold, dt):
    s = metrics(record.y, power, threshold, dt)
    s['threshold_ON_MAE'] = s.pop('on_mae')
    s['power_threshold_F1'] = s.pop('f1')
    s.pop('event_f1_iou05')  # Instantaneous power events are not activity events.
    valid = reference['valid'] & np.isfinite(power)
    active = valid & (reference['labels'] == 1)
    inactive = valid & (reference['labels'] == 0)
    nonnegative = np.maximum(power, 0)
    errors = np.abs(record.y-nonnegative)
    s.update(active_n=int(active.sum()), inactive_n=int(inactive.sum()),
        activity_scored_n=int(valid.sum()),
        activity_support_MAE=float(errors[valid].mean()) if valid.any() else None,
        inactive_MAE=float(errors[inactive].mean()) if inactive.any() else None,
        active_fraction=float(active.sum()/valid.sum()) if valid.any() else None,
        inactive_predicted_wh=float(nonnegative[inactive].sum()*dt/3600),
        inactive_true_wh=float(record.y[inactive].sum()*dt/3600),
        inactive_overprediction_wh=float(np.maximum(nonnegative[inactive]-record.y[inactive], 0).sum()*dt/3600),
        inactive_observed_hours=float(inactive.sum()*dt/3600),
        absolute_energy_error_wh=abs(s['predicted_energy_wh']-s['energy_wh']),
        threshold_ON_n=int((np.isfinite(power) & np.isfinite(record.y) & (record.y > threshold)).sum()))
    s['active_MAE'] = float(np.abs(record.y[active]-np.maximum(power[active], 0)).mean()) if active.any() else None
    s['prediction_coverage'] = float(np.isfinite(power).mean())
    s['observed_target_coverage'] = float(valid.sum() / max(1, np.isfinite(record.y).sum()))
    s['activity_F1'] = s['activity_recall'] = s['activity_false_positive_rate'] = None
    s['complete_activity_recall'] = s['activity_boundary_MAE_seconds'] = None
    s['complete_activity_count'] = 0
    if activity_probability is not None:
        m = valid & np.isfinite(activity_probability)
        truth, pred = reference['labels'][m] == 1, activity_probability[m] >= .5
        tp, fp, fn, tn = [int(v.sum()) for v in (truth & pred, ~truth & pred, truth & ~pred, ~truth & ~pred)]
        s.update(activity_F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                 activity_recall=tp/(tp+fn) if tp+fn else None,
                 activity_false_positive_rate=fp/(fp+tn) if fp+tn else None,
                 activity_tp=tp, activity_fp=fp, activity_fn=fn, activity_tn=tn,
                 predicted_active_fraction=float(pred.mean()) if len(pred) else None,
                 activity_probability_n=int(m.sum()))
        from sklearn.metrics import average_precision_score
        s['activity_PR_AUC'] = float(average_precision_score(truth, activity_probability[m])) if truth.any() else None
        from .data import ranges
        # Assess complete boundaries only where prediction and truth are observed.
        opportunity = m.copy()
        for item in reference['activities']:
            a, b = item['core_start_index'], item['core_end_index']
            complete = (not item['left_censored'] and not item['right_censored']
                        and a > 0 and b < len(m) and m[a-1:b+1].all())
            if not complete:
                opportunity[max(0, a-1):min(len(m), b+1)] = False
        events = [(a, b) for a, b in ranges((activity_probability >= .5) & opportunity)
                  if a > 0 and b < len(m) and opportunity[a-1:b+1].all()]
        truths = [(a['core_start_index'], a['core_end_index']) for a in reference['activities']
                  if not a['left_censored'] and not a['right_censored']
                  and a['core_start_index'] > 0 and a['core_end_index'] < len(m)
                  and opportunity[a['core_start_index']-1:a['core_end_index']+1].all()]
        candidates = []
        for i, (a, b) in enumerate(truths):
            for j, (c, d) in enumerate(events):
                intersection = max(0, min(b, d)-max(a, c))
                iou = intersection / (max(b, d)-min(a, c))
                if iou >= .5:
                    candidates.append((iou, i, j))
        ti, pi, errors = set(), set(), []
        for _, i, j in sorted(candidates, reverse=True):
            if i not in ti and j not in pi:
                ti.add(i); pi.add(j)
                errors.extend([abs(truths[i][q]-events[j][q])*dt for q in (0, 1)])
        s['complete_activity_count'] = len(truths)
        s['complete_predicted_activity_count'] = len(events)
        s['complete_activity_matches'] = len(ti)
        s['event_opportunity_points'] = int(opportunity.sum())
        s['complete_activity_precision'] = len(ti)/len(events) if events and truths else None
        s['complete_activity_F1'] = 2*len(ti)/(len(truths)+len(events)) if truths else None
        s['complete_activity_recall'] = len(ti)/len(truths) if truths else None
        s['activity_boundary_MAE_seconds'] = float(np.mean(errors)) if errors else None
    s.update(record_id=record.id, dataset=record.dataset, house=record.house,
             activity_reference='original_detector_pseudo_labels')
    return s
