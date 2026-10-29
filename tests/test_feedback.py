from fraudscope.feedback import evaluate_feedback

def test_feedback_maturity_and_missing_labels(api,event):
    c,h,db=api
    c.post('/v1/score',headers=h,json=event)
    c.post('/v1/score',headers=h,json=event|{'transaction_id':'second','event_time':200.})
    c.post('/v1/feedback',headers=h,json={'transaction_id':'tx_1','label':1,'available_time':300000.,'source':'confirmed_chargeback'})
    early=evaluate_feedback(db,200000.)
    assert early['mature_events']==2 and early['observed_labels']==0
    later=evaluate_feedback(db,400000.)
    assert later['mature_events']==2 and later['observed_labels']==1 and later['label_coverage']==.5
    assert next(iter(later['by_model'].values()))['status']=='insufficient_class_variation'

