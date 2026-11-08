"""PIT-scoped adapters for legacy generated cross-sectional calculators.

Only nested cross-sectional helpers and ordinal double-argsort are rewritten.
The original formulas and constants remain in their source modules. Nonmembers
cannot change a listed stock's normalization, and ties have average ranks.
"""
from __future__ import annotations
import ast
import copy
import inspect
import textwrap

import numpy as np
from scipy.stats import rankdata


def member_mask(values,panel):
    array=np.asarray(values)
    member=(panel['listing_age']>=0)&~panel['delisted_mask']
    if array.ndim==2 and array.shape==member.shape:
        return np.where(member,array,np.nan)
    return values


def average_ordinals(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2:raise ValueError('cross-sectional rank requires date-by-stock data')
    return rankdata(np.where(np.isfinite(values),values,np.nan),axis=1,method='average',nan_policy='omit')-1.


def _argsort(node):
    return (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='np'
            and node.func.attr=='argsort' and len(node.args)==1
            and all(k.arg in ('axis','kind') for k in node.keywords)
            and all(k.arg!='axis' or (isinstance(k.value,ast.Constant) and k.value.value in (-1,1)) for k in node.keywords))


HELPERS={'_rank_norm','_zscore','_pct_rank','_fillna_median','_rank','_percentile_rank_or_neutral'}


class _ScopeRewrite(ast.NodeTransformer):
    def __init__(self,namespace):self.in_calculator=False;self.in_helper=False;self.namespace=namespace

    def visit_FunctionDef(self,node):
        if node.name=='calc_batch':
            prior=self.in_calculator;self.in_calculator=True
            defined={n.name for n in ast.walk(node) if isinstance(n,ast.FunctionDef)}
            called={n.func.id for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
            helpers=[]
            for name in sorted((called&HELPERS)-defined):
                function=self.namespace.get(name)
                if inspect.isfunction(function):helpers.append(ast.parse(textwrap.dedent(inspect.getsource(function))).body[0])
            node.body=helpers+node.body
            result=self.generic_visit(node);self.in_calculator=prior;return result
        if self.in_calculator and node.name in HELPERS and node.args.args:
            prior=self.in_helper;self.in_helper=True
            node=self.generic_visit(node);self.in_helper=prior
            argument=node.args.args[0].arg
            statement=ast.Assign(targets=[ast.Name(argument,ast.Store())],
                value=ast.Call(ast.Name('__pit_member_mask',ast.Load()),[ast.Name(argument,ast.Load()),ast.Name('panel',ast.Load())],[]))
            node.body.insert(0,statement)
            return node
        return self.generic_visit(node)

    def visit_Return(self,node):
        node=self.generic_visit(node)
        if self.in_helper and node.value is not None:
            node.value=ast.Call(ast.Name('__pit_member_mask',ast.Load()),[node.value,ast.Name('panel',ast.Load())],[])
        return node

    def visit_Call(self,node):
        if _argsort(node) and _argsort(node.args[0]):
            argument=self.visit(node.args[0].args[0])
            return ast.copy_location(ast.Call(ast.Name('__pit_average_ordinals',ast.Load()),[argument],[]),node)
        return self.generic_visit(node)


def adapt_calculator(implementation):
    tree=ast.parse(textwrap.dedent(inspect.getsource(implementation)))
    namespace=dict(vars(inspect.getmodule(implementation)))
    tree=_ScopeRewrite(namespace).visit(copy.deepcopy(tree));ast.fix_missing_locations(tree)
    namespace.update(__pit_member_mask=member_mask,__pit_average_ordinals=average_ordinals)
    # A double underscore in a class body is name-mangled by Python.
    namespace[f'_{implementation.__name__.lstrip("_")}__pit_member_mask']=member_mask
    namespace[f'_{implementation.__name__.lstrip("_")}__pit_average_ordinals']=average_ordinals
    exec(compile(tree,inspect.getsourcefile(implementation),'exec'),namespace)
    return namespace[implementation.__name__]
