"""Synthetic classroom telemetry + transparent rule/ML demonstration.
All data are synthetic; this is not a real environmental sensor system.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

FEATURES = ['temperature_c', 'humidity_pct', 'light_lux']
SEED = 42


def make_normal(n, rng):
    # Educational synthetic assumptions, NOT classroom measurements.
    z = rng.normal(size=n)
    return pd.DataFrame({
        'temperature_c': np.clip(27.5 + 1.0*z + rng.normal(0,.55,n), 23,32),
        'humidity_pct': np.clip(61 + 3.0*z + rng.normal(0,4,n), 43,76),
        'light_lux': np.clip(450 + rng.normal(0,75,n), 260,700),
    })


def make_anomalies(n, rng):
    base = make_normal(n, rng)
    # Groups: hot; humid/dim; unexpected cool+humid; dim lighting.
    modes = rng.integers(0,4,n)
    for idx, mode in enumerate(modes):
        if mode == 0:
            base.loc[idx,'temperature_c'] = rng.uniform(32,36)
        elif mode == 1:
            base.loc[idx,'humidity_pct'] = rng.uniform(80,91)
            base.loc[idx,'light_lux'] = rng.uniform(100,240)
        elif mode == 2:
            base.loc[idx,'temperature_c'] = rng.uniform(22,24)
            base.loc[idx,'humidity_pct'] = rng.uniform(72,88)
        else:
            base.loc[idx,'light_lux'] = rng.uniform(70,210)
    return base


def rule_predict(df):
    # Demo thresholds, NOT adopted classroom comfort standards.
    bad = ((df['temperature_c'] > 30) |
           (df['humidity_pct'] > 75) |
           (df['light_lux'] < 250))
    return bad.astype(int).to_numpy()


def build_experiment():
    rng = np.random.default_rng(SEED)
    train = make_normal(700, rng)
    normal_test = make_normal(240, rng)
    anomaly_test = make_anomalies(60, rng)
    normal_test['ground_truth'] = 0
    anomaly_test['ground_truth'] = 1
    test = pd.concat([normal_test, anomaly_test], ignore_index=True)
    test = test.iloc[rng.permutation(len(test))].reset_index(drop=True)
    model = IsolationForest(n_estimators=150, contamination=.04, random_state=SEED)
    model.fit(train[FEATURES])  # IMPORTANT: only assumed-normal training data
    test['rule_alert'] = rule_predict(test)
    test['ml_alert'] = (model.predict(test[FEATURES]) == -1).astype(int)
    test['anomaly_score'] = -model.decision_function(test[FEATURES])
    metrics = []
    for label,col in [('Rules','rule_alert'), ('Isolation Forest','ml_alert')]:
        y=test['ground_truth'].to_numpy(); p=test[col].to_numpy()
        tn,fp,fn,tp = confusion_matrix(y,p,labels=[0,1]).ravel()
        metrics.append({'method':label,'precision':precision_score(y,p,zero_division=0),
                        'recall':recall_score(y,p,zero_division=0),
                        'f1':f1_score(y,p,zero_division=0),
                        'false_positive_rate':fp/(fp+tn),
                        'TP':int(tp),'FP':int(fp),'TN':int(tn),'FN':int(fn)})
    return model,train,test,pd.DataFrame(metrics)


def demo_recommendations(temp,humidity,light):
    notes=[]
    if temp>30: notes.append('Check AC operation and ventilation; verify with a physical sensor.')
    if humidity>75: notes.append('Check ventilation and possible humidity sources.')
    if light<250: notes.append('Check classroom lighting and available natural light.')
    if not notes: notes.append('Maintain routine environmental monitoring.')
    return notes
