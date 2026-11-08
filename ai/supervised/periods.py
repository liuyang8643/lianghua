"""Explicit data, OOF and observation boundaries for research protocols."""
def data_end(config):
    if config.get('data_end'):
        return config['data_end']
    ranges=[config.get(name) for name in ('train','validation','test')]
    return max(value[1] for value in ranges if value)

def oof_interval(config):
    years=config['fold_years']
    if not years or years!=sorted(set(years)):
        raise ValueError('fold years must be nonempty, unique and increasing')
    return [f'{years[0]}-01-01',f'{years[-1]}-12-31']

def observation_splits(config):
    names=config.get('observation_splits',['oof','fit','internal_validation','train','validation','test'])
    return [name for name in names if name=='oof' or config.get(name)]

def interval(config,split):
    return oof_interval(config) if split=='oof' else config[split]
