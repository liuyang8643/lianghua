"""Opt-in complete local research catalogue, independent of production defaults.

Legacy financial consumers receive strict-announcement PershareIndex fields
through explicit pit_* inputs. Formula names and rewritten versions remain in
the inventory; no successful or failed candidate is silently omitted.
"""
import ast, importlib, inspect, hashlib
from dataclasses import replace
from pathlib import Path
from factor.base import FactorDefinition,FactorMetadata
from factor.registry import PRODUCTION_FACTORS
from offline_data.volume_units import canonical_volume
from factor.library.causal_cross_section import adapt_calculator

UNVERSIONED={'eps','roe','profit_yoy','revenue_yoy','operating_cf_ps','gross_margin','bps'}

def generic_definition(cls, fields, lag=()):
    return FactorDefinition(FactorMetadata(name=cls.__name__,version='local-ic-audit-v1',
        hist_days=int(getattr(cls,'hist_days',0)),required_fields=tuple(sorted(fields)),
        implementation_hash=hashlib.sha256(inspect.getsource(inspect.getmodule(cls)).encode()).hexdigest(),lagged_fields=tuple(lag)),cls)

def plain_catalog(source, runtime_fields):
    selected={x.metadata.name:x for x in PRODUCTION_FACTORS};excluded=[]
    for mod,attr in [('factor.library.bilibili_attention','DEFINITIONS'),('factor.library.bilibili_events','RESEARCH_FACTOR_DEFINITIONS'),
                     ('factor.library.bilibili_price_patterns','DEFINITIONS'),('factor.library.bilibili_513','DEFINITION'),
                     ('factor.library.bilibili_smallcap','SMALLCAP_FACTOR_DEFINITIONS')]:
        entries=getattr(importlib.import_module(mod),attr)
        for entry in entries if isinstance(entries,tuple) else (entries,):selected.setdefault(entry.metadata.name,entry)
    styles=importlib.import_module('factor.library.styles')
    for name,cls in inspect.getmembers(styles,inspect.isclass):
        if cls.__module__!=styles.__name__ or name in selected:continue
        text=inspect.getsource(cls);tree=ast.parse(text)
        fields={v.value for v in ast.walk(tree) if isinstance(v,ast.Constant) and isinstance(v.value,str)}&runtime_fields
        selected[name]=generic_definition(cls,fields)
    for path in sorted((source/'factor_db/factors').glob('*.py')):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and not n.name.startswith('_')]
        strings={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        reason=None
        if path.stem=='deep_value':
            for entry in deep_value_definitions():selected.setdefault(entry.metadata.name,entry)
            continue
        if path.stem=='DividendQuality':reason='requires independently dated dividend events; materialize this data before IC'
        elif path.stem=='IndustryPeerReversalStrict':reason='sealed runtime has no PIT industry_id'
        elif path.stem in ('filter','LiquidityFilter','AmountCeilingFilter','MarketCapFloorFilter','MarketCapCeilingFilter'):
            reason='eligibility filter (1/NaN), not a cross-sectional varying score; standalone IC undefined'
        if reason:
            for n in nodes:
                if n.name not in selected:excluded.append(dict(name=n.name,reason=reason,source=str(path)))
            continue
        module=importlib.import_module('factor_db.factors.'+path.stem)
        for n in nodes:
            if n.name in selected:continue
            if n.name in ('TrendSellRiskFilter','TrendHardDrawdownFilter'):
                excluded.append(dict(name=n.name,reason='eligibility filter 1/NaN; standalone cross-sectional IC undefined',source=str(path)))
                continue
            cls=getattr(module,n.name)
            if not hasattr(cls,'calc_batch'):continue
            # Include fields used by imported helpers by passing the full panel;
            # module-local known fields determine float64 conversion.
            fields=strings&runtime_fields
            lag=('total_share',) if path.stem=='PreCloseMarketCap' else ()
            if strings&UNVERSIONED:
                original=adapt_calculator(cls);financial=tuple(sorted(strings&UNVERSIONED))
                def calc(self,panel,original=original,financial=financial):
                    view=dict(panel)
                    import numpy as np
                    member=(panel['listing_age']>=0)&~panel['delisted_mask']
                    for key in financial:view[key]=np.where(member,panel['pit_'+key],np.nan)
                    return original().calc_batch(view)
                adapted=type(n.name,(),{'calc_batch':calc,'hist_days':getattr(cls,'hist_days',0)})
                definition=generic_definition(original,fields,lag)
                identity=hashlib.sha256((definition.metadata.implementation_hash+Path(__file__).read_text(encoding='utf-8')+inspect.getsource(inspect.getmodule(adapt_calculator))).encode()).hexdigest()
                selected[n.name]=FactorDefinition(replace(definition.metadata,
                    version='legacy-formula-pit-members-average-ties-v2',
                    required_fields=tuple(sorted((fields-UNVERSIONED)|{'listing_age','delisted_mask'}|{'pit_'+key for key in financial})),
                    implementation_hash=identity),adapted)
            else:
                selected[n.name]=generic_definition(cls,fields,lag)
    # The published PB implementation predates the WAN-share unit correction.
    # Correct the research definition without altering a live production default.
    old=selected.get('PBBelowTwoROEAbove10Signal')
    if old is not None:
        def corrected_pb(self,panel):
            view=dict(panel);view['total_share']=panel['total_share']*10000.
            return old.implementation().calc_batch(view)
        cls=type('PBBelowTwoROEAbove10Signal',(),{'calc_batch':corrected_pb})
        selected['PBBelowTwoROEAbove10Signal']=FactorDefinition(replace(old.metadata,
            version='strict-pit-pb-wan-shares-corrected-v1',
            implementation_hash=hashlib.sha256((old.metadata.implementation_hash+Path(__file__).read_text(encoding='utf-8')).encode()).hexdigest()),cls)
    from factor.optimized import OPTIMIZED_FACTORS
    for entry in OPTIMIZED_FACTORS:
        key=entry.metadata.name
        if key in selected:key='Optimized__'+key
        selected[key]=replace(entry,metadata=replace(entry.metadata,name=key))
    for name,entry in list(selected.items()):
        if 'volume' in entry.metadata.required_fields and not entry.raw_runtime_view:
            original=entry.implementation
            def normalized_volume_calc(self,panel,original=original):
                view=dict(panel);view['volume']=canonical_volume(panel,shares=False)
                return original().calc_batch(view)
            cls=type(name,(),{'calc_batch':normalized_volume_calc})
            digest=hashlib.sha256((entry.metadata.implementation_hash+inspect.getsource(inspect.getmodule(canonical_volume))).encode()).hexdigest()
            selected[name]=FactorDefinition(replace(entry.metadata,
                version=entry.metadata.version+'-canonical-volume-lots',
                required_fields=tuple(sorted(set(entry.metadata.required_fields)|{'amount','high','low'})),
                implementation_hash=digest),cls)
    return list(selected.values()),excluded

def bound_groups(runtime):
    # Own the numeric transformation inside the factor preparation boundary.
    # The returned definitions still validate the caller's original raw view.
    from dataclasses import replace
    import numpy as np
    volume=canonical_volume(runtime.data,shares=False);volume.flags.writeable=False
    calculation_runtime=replace(runtime,data={**runtime.data,'volume':volume})
    def wrap(group):
        result=[]
        for entry in group:
            if 'volume' not in entry.metadata.required_fields:
                result.append(entry);continue
            original=entry.implementation
            required=tuple(sorted(set(entry.metadata.required_fields)|{'amount','high','low'}))
            def calc(self,panel,original=original,required=required):
                for name in required:
                    if panel[name] is not runtime.field(name) or panel[name].flags.writeable:
                        raise ValueError('normalized bound factor requires its original immutable inputs')
                return original().calc_batch({**calculation_runtime.data,'trade_dates':runtime.trade_dates,'stock_codes':runtime.stock_codes})
            cls=type(entry.metadata.name,(),{'calc_batch':calc})
            digest=hashlib.sha256((entry.metadata.implementation_hash+inspect.getsource(inspect.getmodule(canonical_volume))).encode()).hexdigest()
            result.append(FactorDefinition(replace(entry.metadata,required_fields=required,
                version=entry.metadata.version+'-canonical-volume-lots',implementation_hash=digest),cls,raw_runtime_view=True))
        return tuple(result)
    from factor.library.bilibili import prepare_bilibili_factor_definitions
    yield wrap(prepare_bilibili_factor_definitions(calculation_runtime))
    from factor.library.bilibili_technical import prepare_technical_definitions
    yield wrap(prepare_technical_definitions(calculation_runtime))
    from env.legality import compute_limit_up_matrix
    limits=compute_limit_up_matrix({**runtime.data,'trade_dates':runtime.trade_dates,'stock_codes':runtime.stock_codes})
    from factor.library.bilibili_supplemental import prepare_supplemental_definitions
    yield wrap(prepare_supplemental_definitions(calculation_runtime,limits))
    from factor.library.bilibili_price_patterns import prepare_bullish_broken_board
    from env.legality import classify_board_types,BOARD_MAIN,BOARD_CYB
    import numpy as np
    boards=classify_board_types(runtime.stock_codes)
    ordinary=np.broadcast_to(boards==BOARD_MAIN,(runtime.n_dates,runtime.n_stocks)).copy()
    ordinary|=(runtime.trade_dates[:,None]<np.datetime64('2020-08-24'))&(boards[None,:]==BOARD_CYB)
    yield (prepare_bullish_broken_board(runtime,limits,ordinary),)
    del limits,ordinary
    from dataclasses import replace
    from factor.library.attention_transition_research import TransitionFeatures,candidate_specs,array_definition
    features=TransitionFeatures(runtime.data)
    for spec in candidate_specs():
        if spec['name']=='attention20':continue # exact alias of BiliLowReturnAttention20
        values=features.calculate(spec)
        yield (replace(array_definition(spec['name'],values),raw_runtime_view=True),)

def financial_adaptations():
    # Exact formulae in recorded video_financial.py, evaluated on the sealed
    # announcement-as-of primitives rather than the unsafe legacy eps/roe fields.
    import numpy as np
    from factor.library.video_financial import _ratio
    fields=('open','total_share','financial_profit_ttm','financial_equity')
    def make(name):
        class RecordedFinancial:
            def calc_batch(self,panel):
                cap=panel['open']*(panel['total_share']*10000.);profit=panel['financial_profit_ttm'];equity=panel['financial_equity']
                cap=np.where((panel['open']>0)&(panel['total_share']>0),cap,np.nan)
                pe=_ratio(cap,profit);pb=_ratio(cap,equity);roe=_ratio(profit,equity);pr=_ratio(pe,roe*100)
                if name=='PRBelowOne':return np.where(pr<1,-pr,np.nan)
                if name=='LowPRDecile':return -pr
                if name=='PBBelowTwoROEAbove10':return np.where((pb<2)&(roe>.1),-pb,np.nan)
                if name=='HighROEDecile':return roe
                limit={'ROEAbove10':.1,'ROEBelow10':.1,'ROEAbove15':.15,'ROEAbove20':.2}[name]
                return np.where(roe<limit if name=='ROEBelow10' else roe>limit,roe,np.nan)
        return FactorDefinition(FactorMetadata(name=name,version='recorded-video-formula-on-sealed-pit-primitives-v1',hist_days=1,required_fields=fields,lagged_fields=('total_share',),implementation_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),RecordedFinancial)
    return tuple(make(n) for n in ('PRBelowOne','LowPRDecile','PBBelowTwoROEAbove10','ROEAbove10','ROEBelow10','ROEAbove15','ROEAbove20','HighROEDecile'))


def deep_value_definitions():
    """Same historical formulas with no hidden unversioned snapshot read."""
    from factor.library import deep_value_pit
    fields=('open','st_mask','listing_age','delisted_mask','pit_bps','pit_eps','pit_operating_cf_ps','pit_roe','pit_profit_yoy')
    return tuple(generic_definition(getattr(deep_value_pit,name),fields)
                 for name in deep_value_pit.NAMES)
