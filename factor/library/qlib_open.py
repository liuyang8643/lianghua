"""Qlib Alpha158/360 expressions adapted to the decision open.

Original expression catalogue: Microsoft Qlib, MIT license, pinned in the
adjacent JSON. Every original expression is evaluated on completed bars (T-1).
This module neither fetches data nor selects a stock universe. Native Qlib
operator semantics are retained; price-adjustment gaps are explicitly masked.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from numba import njit
import pandas as pd

from factor.base import FactorDefinition, FactorMetadata
from offline_data.volume_units import canonical_volume

CATALOG_PATH = Path(__file__).with_name('qlib_catalog.json')
CATALOG = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
VERSION = 'qlib-open-completed-bar-v4-explicit-volume-units'


def lag(values: np.ndarray, periods: int) -> np.ndarray:
    if periods < 0:
        raise ValueError('future references are forbidden in features')
    if periods == 0:
        return values
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if periods < len(values):
        out[periods:] = values[:-periods]
    return out


@njit(cache=True, fastmath=False)
def _economic_state(close, preclose):
    """Forward accumulation uses only information available by each row.

    An unknown bridge starts a new scale segment; it is not assumed continuous.
    Callers mask features crossing such a bridge through missing price rows.
    """
    out = np.ones(close.shape, dtype=np.float64)
    segments = np.full(close.shape, -1, dtype=np.int32)
    for j in range(close.shape[1]):
        scale = np.nan; segment = -1
        for t in range(len(close)):
            c = close[t-1,j] if t>0 else np.nan
            p = preclose[t,j]
            if segment>=0 and t>0 and np.isfinite(c) and c > 0 and np.isfinite(p) and p > 0:
                scale *= c / p
            elif np.isfinite(close[t,j]) and close[t,j]>0:
                scale = 1.0; segment = t
            else:
                scale = np.nan; segment = -1
            out[t, j] = scale
            segments[t,j] = segment
    return out,segments


def economic_adjustment(close,preclose):
    return _economic_state(close,preclose)[0]


@njit(cache=True, fastmath=False)
def _regression(x, window, which):
    result = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        for t in range(x.shape[0]):
            start = max(0, t-window+1)
            n = 0; sx = 0.; sy = 0.
            for k in range(start, t+1):
                if np.isfinite(x[k, j]):
                    n += 1; sx += k-start; sy += x[k, j]
            if n < 2:
                continue
            mx, my = sx/n, sy/n
            xx = 0.; yy = 0.; xy = 0.
            for k in range(start, t+1):
                if np.isfinite(x[k, j]):
                    dx, dy = k-start-mx, x[k, j]-my
                    xx += dx*dx; yy += dy*dy; xy += dx*dy
            if xx > 0:
                slope = xy/xx
                if which==0: result[t,j] = slope
                elif which==2: result[t,j] = x[t, j] - (my + slope*(t-start-mx))
                elif yy > 0 and np.sqrt(yy/(n-1)) > 2e-5:
                    result[t,j] = xy*xy/(xx*yy)
    return result


@njit(cache=True, fastmath=False)
def _rolling_index(x, window, maximum):
    out = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        for t in range(len(x)):
            start = max(0, t-window+1)
            best = np.nan; pos = -1; first_nan = -1
            for k in range(start, t+1):
                v = x[k, j]
                if not np.isfinite(v) and first_nan<0: first_nan=k-start+1
                if np.isfinite(v) and (pos < 0 or (v > best if maximum else v < best)):
                    best, pos = v, k-start+1
            if pos > 0:
                # Qlib calls ndarray.argmax/argmin on the full rolling slice:
                # a NaN wins; an entirely missing window has no min_periods.
                out[t, j] = first_nan if first_nan>0 else pos
    return out


def _lookback(node):
    """Inclusive nearest/farthest historical row used by an expression."""
    if isinstance(node,ast.Name): return (0,0)
    if isinstance(node,ast.Constant): return None
    if isinstance(node,ast.Call):
        name=node.func.id
        if name=='Ref':
            near,far=_lookback(node.args[0]); n=int(ast.literal_eval(node.args[1]))
            return near+n,far+n
        rolling={'Mean','Std','Sum','Max','Min','Quantile','Rank','Slope','Rsquare','Resi','IdxMax','IdxMin'}
        if name in rolling:
            near,far=_lookback(node.args[0]); return near,far+int(ast.literal_eval(node.args[1]))-1
        if name=='Corr':
            a,b=_lookback(node.args[0]),_lookback(node.args[1])
            return min(a[0],b[0]),max(a[1],b[1])+int(ast.literal_eval(node.args[2]))-1
        children=node.args
    else:
        children=[child for child in ast.iter_child_nodes(node) if isinstance(child,ast.expr)]
    bounds=[bound for child in children if (bound:=_lookback(child)) is not None]
    return (min(x[0] for x in bounds),max(x[1] for x in bounds)) if bounds else None


class ExpressionPanel:
    """Restricted expression evaluator with bounded subexpression caching."""
    def __init__(self, fields: dict[str, np.ndarray], *, cache_entries: int = 5):
        self.fields = fields
        self.cache_entries = cache_entries
        self.cache = OrderedDict()

    def evaluate(self, expression: str) -> np.ndarray:
        rewritten = re.sub(r'\$([A-Za-z_]+)', r'FIELD_\1', expression)
        tree=ast.parse(rewritten,mode='eval').body
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            result = self._node(tree)
        out = np.asarray(result, dtype=np.float64)
        # Inf is not an observed finite factor score.
        if np.isinf(out).any():
            out = out.copy(); out[np.isinf(out)] = np.nan
        if '__segment_start__' in self.fields:
            nearest,farthest=_lookback(tree)
            segment=self.fields['__segment_start__']
            rows=np.arange(len(out))
            latest=np.maximum(0,rows-nearest)
            valid=(rows[:,None]>=farthest)&(segment[latest]>=0)&(segment[latest]<=rows[:,None]-farthest)
            out=np.where(valid,out,np.nan)
        return out

    def _node(self, node):
        key = ast.dump(node, include_attributes=False)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id.startswith('FIELD_'):
            return self.fields[node.id[6:]]
        if isinstance(node, ast.BinOp):
            functions = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                         ast.Div: np.divide, ast.Pow: np.power}
            if type(node.op) not in functions:
                raise ValueError('unsupported binary operation')
            result = functions[type(node.op)](self._node(node.left), self._node(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            result = -self._node(node.operand)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = self._node(node.left), self._node(node.comparators[0])
            functions = {ast.Gt: np.greater, ast.Lt: np.less, ast.GtE: np.greater_equal,
                         ast.LtE: np.less_equal, ast.Eq: np.equal}
            result = functions[type(node.ops[0])](a,b).astype(np.float64)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            result = self._call(node.func.id, [self._node(x) for x in node.args])
        else:
            raise ValueError(f'unsupported feature syntax: {ast.dump(node)}')
        if isinstance(result, np.ndarray) and self.cache_entries:
            self.cache[key] = result
            while len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)
        return result

    def _call(self, name, args):
        if name == 'Ref':
            return lag(args[0], int(args[1]))
        if name in ('Greater', 'Less'):
            return (np.maximum if name == 'Greater' else np.minimum)(*args)
        if name in ('Abs', 'Log'):
            return (np.abs if name == 'Abs' else np.log)(args[0])
        x = np.asarray(args[0], dtype=np.float64)
        window = int(args[-1] if name == 'Corr' else args[1])
        if window <= 0:
            raise ValueError('positive fixed rolling windows are required')
        if name in ('Slope', 'Rsquare', 'Resi'):
            return _regression(x, window, ('Slope', 'Rsquare', 'Resi').index(name))
        if name in ('IdxMax', 'IdxMin'):
            return _rolling_index(x, window, name == 'IdxMax')
        roll = pd.DataFrame(x).rolling(window, min_periods=1)
        methods = {'Mean':'mean', 'Std':'std', 'Sum':'sum', 'Max':'max', 'Min':'min'}
        if name in methods:
            return getattr(roll, methods[name])().to_numpy()
        if name == 'Quantile':
            return roll.quantile(float(args[2])).to_numpy()
        if name == 'Rank':
            return roll.rank(pct=True).to_numpy()
        if name == 'Corr':
            y = pd.DataFrame(args[1]); other = y.rolling(window, min_periods=1)
            result = roll.corr(y).to_numpy()
            result[np.isclose(roll.std().to_numpy(), 0, atol=2e-5)
                   | np.isclose(other.std().to_numpy(), 0, atol=2e-5)] = np.nan
            return result
        raise ValueError(f'unsupported Qlib operator: {name}')


def price_fields(panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize mixed local lots/shares to shares; amount is yuan.

    Economic adjustment follows the local close/preClose bridge. Volumes are
    transformed inversely so price-times-volume remains economically coherent.
    The chosen anchor uses no future observation.
    """
    scale,segments = _economic_state(panel['close'], panel['preClose'])
    fields = {name: np.asarray(panel[name], dtype=np.float64)*scale for name in ('open','high','low','close')}
    volume = canonical_volume(panel,shares=True)
    fields['volume'] = volume/scale
    fields['vwap'] = np.divide(panel['amount'], volume, out=np.full(volume.shape,np.nan), where=volume > 0)*scale
    fields['__segment_start__'] = segments
    return fields


def definitions() -> tuple[FactorDefinition, ...]:
    import inspect
    identity = hashlib.sha256(Path(__file__).read_bytes()+CATALOG_PATH.read_bytes()+inspect.getsource(inspect.getmodule(canonical_volume)).encode()).hexdigest()
    result = []
    for item in CATALOG['features']:
        expression = item['expression']
        def calculate(self, panel, expression=expression):
            return ExpressionPanel(price_fields(panel)).evaluate(expression)
        implementation = type(item['name'], (), {'calc_batch': calculate, 'hist_days':61})
        result.append(FactorDefinition(FactorMetadata(item['name'], VERSION, 61,
            ('open','high','low','close','volume','amount','preClose'), identity), implementation))
    return tuple(result)


def iter_features(panel: dict[str, np.ndarray]):
    """One shared input representation; stream columns instead of a 3-D tensor."""
    evaluator = ExpressionPanel(price_fields(panel))
    for item in CATALOG['features']:
        yield item, evaluator.evaluate(item['expression'])
