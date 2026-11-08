"""Readable account/target/difference log. Python 3.6, no strategy/SDK imports."""


def holding_comparison(document, positions, names=None):
    names = names or {}
    actual = {p['stock_code']: int(p['volume']) for p in positions if p['volume'] > 0}
    priority = list(document['buy_priority'])
    priority.extend(sorted((set(actual) | set(document['targets'])) - set(priority)))
    rows = []
    for code in priority:
        current = actual.get(code, 0)
        target = document['targets'].get(code)
        quantity = target['shares'] if target is not None else current
        difference = quantity - current
        action = ('买入 %d' % difference if difference > 0 else
                  '卖出 %d' % -difference if difference < 0 else '不调整')
        if target is None:
            action = '计划未涉及，原样保留'
        rows.append(dict(code=code, name=names.get(code, ''), current=current,
                         target=quantity, difference=difference, action=action,
                         managed=target is not None))
    return rows


def comparison_lines(rows):
    lines = ['账户持仓 vs 策略目标 vs 多退少补（股）',
             '证券代码 名称 | 账户持仓 | 目标持仓 | 多退少补']
    for row in rows:
        lines.append('{code} {name} | {current} | {target} | {action}'.format(**row))
    buys = [r for r in rows if r['difference'] > 0]
    sells = [r for r in rows if r['difference'] < 0]
    lines.append('多退少补合计：买入 %d 只/%d 股，卖出 %d 只/%d 股' % (
        len(buys), sum(r['difference'] for r in buys),
        len(sells), -sum(r['difference'] for r in sells)))
    return lines
