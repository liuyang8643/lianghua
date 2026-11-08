"""下单前持久化执行意图；重启后以券商委托为准恢复，避免重复发单。"""
import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from xtquant import xtconstant


class ExecutionJournal:
    def __init__(self, directory, account):
        self.directory = Path(directory)
        self.account = account
        self.state = None
        self.path = None

    def load(self, trade_date):
        self.path = self.directory / f'execution_{trade_date.isoformat()}.json'
        self.state = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            if self.state['version'] != 1 or self.state['account'] != self.account:
                raise RuntimeError('执行记录账户或版本不一致，禁止下单')
            pending = dict(self.state['pending'])
            for key in ('signal_date', 'trade_date'):
                pending[key] = datetime.strptime(pending[key], '%Y-%m-%d').date()
            if pending['trade_date'] != trade_date:
                raise RuntimeError('执行记录日期不一致')
            return pending
        return None

    def begin(self, pending, *, metadata=None):
        existing = self.load(pending['trade_date'])
        if existing is not None:
            return existing
        # 仅保留执行器消费的数据，不保存临时打分缓存。
        keys = ('signal_date', 'trade_date', 'sell_orders', 'buy_allocations',
                'buy_n_stocks', 'prices', 'limit_prices')
        keys += tuple(key for key in ('max_positions','unmanaged_codes') if key in pending)
        self.state = {'version': 1, 'account': self.account,
                      'id': uuid.uuid4().hex[:12], 'completed': False,
                      'pending': {key: pending[key] for key in keys}, 'requests': []}
        if metadata is not None:
            reserved = set(self.state) & set(metadata)
            if reserved:
                raise ValueError('Journal metadata overwrites reserved fields')
            self.state.update(metadata)
        self.save()
        return pending

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        def encode(value):
            if isinstance(value, date):
                return value.isoformat()
            if hasattr(value, 'item'):
                return value.item()
            raise TypeError(type(value).__name__)
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(self.state, default=encode, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def before_submit(self, direction, code, shares, price):
        request = {'remark': f"wbr:{self.state['id']}:{len(self.state['requests']):06d}",
                   'direction': direction, 'code': code, 'shares': int(shares),
                   'price': price, 'order_id': None}
        self.state['requests'].append(request)
        # 意图不能保存时尚未发出委托；异常必须阻止新增订单。
        self.save()
        return request

    def mark_not_submitted(self, request, reason):
        """Only a durable broker-bridge response can establish this state."""
        if request not in self.state['requests']:
            raise RuntimeError('未找到待终结的执行意图')
        request['submission_state'] = 'not_submitted'
        request['rejection_reason'] = str(reason)
        self.save()

    def recover_orders(self, orders, *, not_submitted=()):
        by_remark = {}
        for order in orders:
            if order.order_remark in by_remark:
                if any(r['remark'] == order.order_remark for r in self.state['requests']):
                    raise RuntimeError('券商存在重复执行标识，禁止自动恢复')
            by_remark[order.order_remark] = order
        recovered = []
        for request in self.state['requests']:
            order = by_remark.get(request['remark'])
            if request['remark'] in not_submitted:
                if order is not None:
                    raise RuntimeError('桥接未提交证明与券商委托冲突，禁止自动重发')
                if request.get('submission_state') != 'not_submitted':
                    self.mark_not_submitted(request, '客户端桥接持久化证明未调用下单接口')
            if request.get('submission_state') == 'not_submitted':
                if order is not None:
                    raise RuntimeError('未提交证明与券商委托冲突，禁止自动重发')
                continue
            if order is None:
                raise RuntimeError(f"委托状态不确定，等待券商确认: {request['remark']}")
            expected_type = xtconstant.STOCK_BUY if request['direction'] == 'BUY' else xtconstant.STOCK_SELL
            if (order.stock_code != request['code'] or int(order.order_volume) != request['shares']
                    or order.order_type != expected_type or int(order.order_id) <= 0):
                raise RuntimeError('券商委托与已保存意图不一致')
            recovered.append((request, order))
        return recovered
