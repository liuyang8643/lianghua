import copy
from datetime import date, datetime
import json
from pathlib import Path

import pytest

from trading.bridge_wire import atomic_json
from trading.daily_records import write_planner_record, write_executor_record, build_executor_record
from trading.daily_notify import deliver_daily


def facts():
    day = '2026-11-03'; stamp = datetime(2026,11,3,9,32).timestamp()
    doc = dict(id='abcdefabcdef',account='test',trade_date=day,buy_priority=['600002.SH'],
               targets={'600001.SH':{'shares':0},'600002.SH':{'shares':100}})
    initial = dict(account='test',trade_date=day,captured_at=stamp-300,asset={},
                   positions=[dict(stock_code='600001.SH',volume=200),dict(stock_code='159001.SZ',volume=100)])
    orders=[]; trades=[]; reqs=[]
    for n,(code,side,volume,filled,status) in enumerate([
        ('600001.SH',24,200,50,53),('600001.SH',24,150,150,56),('600002.SH',23,100,100,56)],1):
        remark='wbr:abcdefabcdef:%06d'%n
        reqs.append(dict(remark=remark,code=code,direction='BUY' if side==23 else 'SELL',shares=volume))
        orders.append(dict(account_id='test',stock_code=code,order_type=side,order_volume=volume,
                           traded_volume=filled,order_status=status,order_time=stamp,order_id=n,order_remark=remark))
        trades.append(dict(account_id='test',stock_code=code,order_type=side,traded_volume=filled,
                           traded_time=stamp,traded_id=str(n),order_id=n,order_remark=remark,traded_price=10.,traded_amount=filled*10.))
    journal=dict(account='test',id='execution123',completed=True,target_document=doc,requests=reqs,
        initial_account=initial,pending=dict(sell_orders=[['600001.SH',200]],buy_allocations={'600002.SH':100}))
    snapshot=dict(account='test',trade_date=day,captured_at=stamp,asset={},
        positions=[dict(stock_code='600002.SH',volume=100),dict(stock_code='159001.SZ',volume=100)],
        orders=orders,trades=trades,trades_available=True)
    return doc,initial,journal,snapshot


def test_two_daily_files_are_immutable_and_preserve_replacement_fills(tmp_path):
    doc,initial,journal,snapshot=facts()
    plan=write_planner_record(tmp_path,doc,snapshot=initial)
    result=write_executor_record(tmp_path,journal,snapshot)
    original=(plan.read_bytes(),result.read_bytes())
    snapshot['asset']={'market_value':999}
    write_planner_record(tmp_path,doc,snapshot=initial)
    write_executor_record(tmp_path,journal,snapshot)
    assert original==(plan.read_bytes(),result.read_bytes())
    report=json.loads(result.read_text())
    assert report['filled_net']=={'600001.SH':-200,'600002.SH':100}
    assert report['remaining']=={'600001.SH':0,'600002.SH':0}
    assert len(report['orders'])==3 and len(report['trades'])==3 and report['state']=='completed'


@pytest.mark.parametrize('bad', ['unavailable','conflict','missing','wrong_account','wrong_order_id','per_order_amounts'])
def test_invalid_completion_cannot_be_archived_as_success(tmp_path,bad):
    doc,initial,journal,snapshot=facts()
    if bad=='unavailable': snapshot['trades_available']=False
    elif bad=='conflict': snapshot['trades'].append(dict(snapshot['trades'][0],traded_amount=1))
    elif bad=='missing': snapshot['trades'].pop()
    elif bad=='wrong_account': snapshot['account']='other'
    elif bad=='wrong_order_id': snapshot['trades'][0]['order_id']=2
    else:
        snapshot['trades'][0]['traded_volume']=75
        snapshot['trades'][1]['traded_volume']=125
    with pytest.raises(ValueError): write_executor_record(tmp_path,journal,snapshot)
    assert not (tmp_path/'daily/executor_2026-11-03.json').exists()


def test_legacy_before_is_explicitly_reconstructed(tmp_path):
    doc,_,journal,_=facts(); del journal['initial_account']
    report=json.loads(write_planner_record(tmp_path,doc,journal=journal).read_text())
    assert report['backfilled'] and report['initial_account'] is None
    assert report['initial_holdings']=={'600001.SH':200,'600002.SH':0}
    assert report['initial_holdings_source'].startswith('reconstructed_')


def test_precall_rejection_and_late_fills_do_not_falsely_fail_final_report():
    _,_,journal,snapshot=facts()
    journal['requests'].append(dict(remark='wbr:abcdefabcdef:999999',code='600002.SH',direction='BUY',
        shares=100,submission_state='not_submitted',rejection_reason='guard before SDK'))
    journal['completed']=False  # Engine ended before late broker facts arrived.
    report=build_executor_record(journal,snapshot)
    assert report['state']=='completed' and report['journal_completed'] is False
    assert len(report['pre_submission_rejections'])==1


def test_precall_rejection_conflicting_with_trade_alone_is_rejected():
    _,_,journal,snapshot=facts()
    remark='wbr:abcdefabcdef:999999'
    journal['requests'].append(dict(remark=remark,code='600001.SH',direction='SELL',shares=50,
                                    submission_state='not_submitted'))
    snapshot['trades'].append(dict(snapshot['trades'][0],order_remark=remark,traded_id='contradiction'))
    with pytest.raises(ValueError,match='Non-submission proof'):
        build_executor_record(journal,snapshot)


class Sender:
    def __init__(self,fail=False): self.calls=[]; self.fail=fail
    def send_card(self,card,**kwargs):
        self.calls.append(('create',card)); return None if self.fail else 'message1'
    def update_card(self,message_id,card):
        assert message_id=='message1'; self.calls.append(('update',card)); return True


def setup(directory):
    atomic_json(directory/'config.json',{'account':'test'})
    atomic_json(directory/'notification_config.json',{'enabled':True})


def test_one_card_updated_once_from_two_json_sources(tmp_path,monkeypatch):
    import trading.day_board as board
    monkeypatch.setattr(board,'get_stock_name_at_date',lambda code,day:code)
    doc,initial,journal,snapshot=facts(); setup(tmp_path)
    write_planner_record(tmp_path,doc,snapshot=initial)
    sender=Sender(); day=date(2026,11,3)
    deliver_daily(tmp_path,day,sender=sender); deliver_daily(tmp_path,day,sender=sender)
    write_executor_record(tmp_path,journal,snapshot)
    deliver_daily(tmp_path,day,sender=sender); deliver_daily(tmp_path,day,sender=sender)
    assert [c[0] for c in sender.calls]==['create','update']
    card=sender.calls[-1][1]
    assert card['header']['template']=='wathet'
    table=next(e for e in card['body']['elements'] if e['tag']=='table')
    assert len(table['columns'])==5
    assert table['rows'][-1]['name']=='— 合计 —'
    assert any('✅ 已成' in r['log'] for r in table['rows'])


def test_unknown_send_does_not_create_duplicate(tmp_path):
    doc,initial,_,_=facts(); setup(tmp_path); write_planner_record(tmp_path,doc,snapshot=initial)
    sender=Sender(fail=True)
    for _ in range(2):
        with pytest.raises(RuntimeError): deliver_daily(tmp_path,date(2026,11,3),sender=sender)
    assert len(sender.calls)==1


def test_simulation_cannot_be_sent(tmp_path):
    doc,initial,_,_=facts(); setup(tmp_path)
    write_planner_record(tmp_path,doc,snapshot=initial,simulated=True)
    sender=Sender()
    with pytest.raises(ValueError): deliver_daily(tmp_path,date(2026,11,3),sender=sender)
    assert not sender.calls


def test_local_archive_recovers_without_feishu_enabled(tmp_path):
    doc,initial,journal,_=facts()
    doc['planning_positions']={p['stock_code']:p['volume'] for p in initial['positions']}
    atomic_json(tmp_path/'config.json',{'account':'test'})
    atomic_json(tmp_path/'targets_2026-11-03.json',doc)
    sender=Sender()
    assert deliver_daily(tmp_path,date(2026,11,3),sender=sender) is None
    report=json.loads((tmp_path/'daily/planner_2026-11-03.json').read_text())
    assert report['initial_holdings']==doc['planning_positions']
    assert report['initial_holdings_source']=='frozen_publisher_positions'
    assert not sender.calls


def test_simulated_target_backfill_stays_simulated_and_cannot_execute(tmp_path):
    from trading.target_protocol import validate_target
    doc,initial,_,_=facts()
    doc.update(simulation=True,planning_positions={})
    setup(tmp_path); atomic_json(tmp_path/'targets_2026-11-03.json',doc)
    sender=Sender()
    with pytest.raises(ValueError,match='Invalid local daily'):
        deliver_daily(tmp_path,date(2026,11,3),sender=sender)
    assert json.loads((tmp_path/'daily/planner_2026-11-03.json').read_text())['simulated']
    with pytest.raises(ValueError,match='Simulation target'):
        validate_target(doc,'test',date(2026,11,3),0)
    assert not sender.calls


def test_final_report_updates_original_message_and_sends_attachments_once(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    import trading.day_board as board
    monkeypatch.setattr(board, 'get_stock_name_at_date', lambda code, day: code)
    doc, initial, journal, snapshot = facts(); setup(tmp_path)
    write_planner_record(tmp_path, doc, snapshot=initial)
    result = json.loads(write_executor_record(tmp_path, journal, snapshot).read_text())
    sender = Sender(); day = date(2026, 11, 3)
    deliver_daily(tmp_path, day, sender=sender)
    sender.send_file = lambda *a, **k: (sender.calls.append(('file', a)) or
        SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id='file1')))
    html = tmp_path / 'report.html'; html.write_text('verified html')
    close = dict(account='test', trade_date=day.isoformat(), target_id=doc['id'],
        execution_id=journal['id'], holdings=result['actual_holdings'],
        summary=dict(live_daily_pnl=-100, live_daily_return_pct=-1, bt_daily_return_pct=-0.9),
        detail_card={'header': {}}, html_path=str(html),
        html_sha256=hashlib.sha256(html.read_bytes()).hexdigest())
    atomic_json(tmp_path / 'daily/close_2026-11-03.json', close)
    for _ in range(2): deliver_daily(tmp_path, day, sender=sender)
    assert [c[0] for c in sender.calls] == ['create', 'update', 'file', 'create']
    assert sender.calls[1][1]['header']['template'] == 'red'
    receipt = json.loads((tmp_path / 'daily/feishu_2026-11-03.json').read_text())
    assert receipt['message_id'] == 'message1' and receipt['phase'] == 'close'


def test_ambiguous_html_send_is_not_retried(tmp_path):
    import hashlib
    from trading.daily_notify import _deliver_close_attachments
    (tmp_path / 'daily').mkdir()
    html = tmp_path / 'report.html'; html.write_text('report')
    packet = dict(html_path=str(html), html_sha256=hashlib.sha256(html.read_bytes()).hexdigest())
    calls = []
    class FailedSender:
        def send_file(self, *a, **k): calls.append(a); return None
    for _ in range(2):
        with pytest.raises(RuntimeError):
            _deliver_close_attachments(tmp_path, '2026-11-03', packet, FailedSender())
    assert len(calls) == 1


def test_success_without_file_message_id_is_unknown(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from trading.daily_notify import _deliver_close_attachments
    (tmp_path / 'daily').mkdir()
    html = tmp_path / 'report.html'; html.write_text('report')
    packet = dict(html_path=str(html), html_sha256=hashlib.sha256(html.read_bytes()).hexdigest())
    sender = SimpleNamespace(send_file=lambda *a, **k:
        SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=None)))
    with pytest.raises(RuntimeError):
        _deliver_close_attachments(tmp_path, '2026-11-03', packet, sender)
    assert json.loads((tmp_path / 'daily/feishu_close_2026-11-03.json').read_text())['html']['state'] == 'unknown'


def test_empty_original_detail_is_skipped(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from trading.daily_notify import _deliver_close_attachments
    (tmp_path / 'daily').mkdir()
    html = tmp_path / 'report.html'; html.write_text('report')
    packet = dict(html_path=str(html), html_sha256=hashlib.sha256(html.read_bytes()).hexdigest(), detail_card=None)
    sender = SimpleNamespace(send_file=lambda *a, **k:
        SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id='file1')))
    for _ in range(2): _deliver_close_attachments(tmp_path, '2026-11-03', packet, sender)
    assert json.loads((tmp_path / 'daily/feishu_close_2026-11-03.json').read_text())['detail']['state'] == 'skipped'


def test_corrected_report_updates_detail_and_replaces_html_once(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from trading.daily_notify import _deliver_close_attachments
    (tmp_path / 'daily').mkdir()
    html = tmp_path / 'report.html'; html.write_text('old report')
    packet = dict(account='test',html_path=str(html),html_sha256=hashlib.sha256(html.read_bytes()).hexdigest(),detail_card={'value':1})
    sender = Sender()
    sender.send_file = lambda *a, **k: (sender.calls.append(('file', k)) or
        SimpleNamespace(success=lambda:True,data=SimpleNamespace(message_id='file1')))
    _deliver_close_attachments(tmp_path,'2026-11-03',packet,sender)
    html.write_text('corrected report')
    packet.update(html_sha256=hashlib.sha256(html.read_bytes()).hexdigest(),detail_card={'value':2})
    for _ in range(2): _deliver_close_attachments(tmp_path,'2026-11-03',packet,sender)
    assert [c[0] for c in sender.calls]==['file','create','file','update']
    assert 'corrected' in sender.calls[2][1]['file_name']
    receipt=json.loads((tmp_path/'daily/feishu_close_2026-11-03.json').read_text())
    assert len(receipt['html']['history'])==1 and receipt['detail']['message_id']=='message1'
