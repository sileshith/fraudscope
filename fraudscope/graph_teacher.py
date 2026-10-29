"""Past-only expanding-window teacher for supervised edge-risk gates."""
import numpy as np
from .models import make_models

def causal_teacher_scores(df,x,numeric,categorical,train_rows,train_boundary,seed=42,time_col='event_time',label_col='is_fraud',lag_days=2,folds=5):
    t=df[time_col].to_numpy(float);y=df[label_col].to_numpy(int)
    available=df.label_available_time.to_numpy(float) if 'label_available_time' in df else t+lag_days*86400
    scores=np.full(len(df),.02,dtype=np.float32);manifests=[];allowed=np.zeros(len(df),bool);allowed[train_rows]=True
    bounds=np.linspace(t[0],train_boundary,folds+1)
    for start,end in zip(bounds[:-1],bounds[1:]):
        target=np.flatnonzero((t>=start)&(t<end));fit=np.flatnonzero(allowed&(available<start))
        record={'score_start':float(start),'score_end':float(end),'fit_rows':len(fit),'scored_rows':len(target),
                'max_fit_label_time':float(available[fit].max()) if len(fit) else None}
        if len(fit)<200 or len(np.unique(y[fit]))<2:
            record['method']='fixed_prior_0.02_no_fitted_teacher'
        else:
            teacher=make_models(numeric,categorical,seed)['logistic'];teacher.fit(x.iloc[fit],y[fit])
            scores[target]=teacher.predict_proba(x.iloc[target])[:,1];record['method']='past_only_logistic'
        manifests.append(record)
    final=make_models(numeric,categorical,seed)['logistic'];final.fit(x.iloc[train_rows],y[train_rows])
    future=np.flatnonzero(t>=train_boundary);scores[future]=final.predict_proba(x.iloc[future])[:,1]
    manifests.append({'score_start':float(train_boundary),'score_end':None,'fit_rows':len(train_rows),'scored_rows':len(future),
                      'max_fit_label_time':float(available[train_rows].max()),'method':'frozen_train_only_logistic'})
    assert all(r['max_fit_label_time'] is None or r['max_fit_label_time']<r['score_start'] for r in manifests)
    return scores,manifests,final
