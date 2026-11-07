from ai.supervised.periods import data_end,oof_interval,observation_splits

def test_expanded_study_has_no_old_test_or_2017_oof_cutoff():
    c={'train':['2005-01-01','2021-12-31'],'fit':['2005-01-01','2020-12-31'],'internal_validation':['2021-01-01','2021-12-31'],'validation':['2022-01-01','2026-10-31'],'test':None,'fold_years':list(range(2011,2022))}
    assert data_end(c)=='2026-10-31'
    assert oof_interval(c)==['2011-01-01','2021-12-31']
    assert observation_splits(c)==['oof','fit','internal_validation','train','validation']

def test_legacy_protocol_retains_old_ranges():
    c={'train':['2005-01-01','2017-12-31'],'validation':['2018-01-01','2021-12-31'],'test':['2022-01-01','2026-10-31'],'fold_years':list(range(2011,2018))}
    assert data_end(c)=='2026-10-31'
    assert oof_interval(c)==['2011-01-01','2017-12-31']
    assert observation_splits(c)==['oof','train','validation','test']
