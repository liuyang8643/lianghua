from datetime import date,datetime
import json
import pytest


def test_crossday_rechecks_same_previous_session_and_prepares_before_open(tmp_path,monkeypatch):
    from trading import planner_watchdog as planner,plan_publisher,preopen,daily_notify
    from utils.stock import time as calendar
    moments=[datetime(2026,11,8,23,59),datetime(2026,11,9,8,30),datetime(2026,11,9,9,20)]
    clock=[0];history=[];prepared=[]
    class Clock:
        @staticmethod
        def now():return moments[clock[0]]
    days=[date(2026,11,6),date(2026,11,9)]
    monkeypatch.setattr(planner,'datetime',Clock)
    monkeypatch.setattr(planner.time,'time',lambda:2000000000+clock[0]*301)
    monkeypatch.setattr(calendar,'_get_trading_calendar_state',lambda:(days,None,date(2026,12,31)))
    monkeypatch.setattr(calendar,'get_last_trading_day',lambda d:max(x for x in days if x<=d))
    monkeypatch.setattr(plan_publisher,'ensure_local_history',lambda d:history.append(d))
    monkeypatch.setattr(preopen,'prepare',lambda d,*a,**k:prepared.append((d,k['allow_calculation'])))
    monkeypatch.setattr(daily_notify,'deliver_daily',lambda *a:None)
    def advance(seconds):
        clock[0]+=1
        if clock[0]==len(moments):raise SystemExit('finished test')
    monkeypatch.setattr(planner.time,'sleep',advance)
    with pytest.raises(SystemExit,match='finished test'):planner.run('unused',tmp_path)
    assert history==[date(2026,11,6),date(2026,11,6)]
    assert prepared==[(date(2026,11,9),True)]


def test_missing_preopen_cache_never_triggers_late_history_calculation(tmp_path,monkeypatch):
    from trading import preopen
    monkeypatch.setattr(preopen,'_READY',None)
    monkeypatch.setattr(preopen,'identity',lambda *a:dict(day='2026-11-09'))
    monkeypatch.setattr(preopen,'load_runtime_slice',lambda *a,**k:pytest.fail('late full history load'))
    with pytest.raises(RuntimeError,match='preflight is not ready'):
        preopen.prepare(date(2026,11,9),'unused',tmp_path,allow_calculation=False)


def test_a_changed_input_identity_cannot_reuse_the_old_day_cache(tmp_path,monkeypatch):
    from trading import preopen
    from types import SimpleNamespace
    monkeypatch.setattr(preopen,'_READY',SimpleNamespace(identity=dict(day='2026-11-09',inputs='old')))
    monkeypatch.setattr(preopen,'identity',lambda *a:dict(day='2026-11-09',inputs='new'))
    with pytest.raises(RuntimeError,match='preflight is not ready'):
        preopen.prepare(date(2026,11,9),'unused',tmp_path,allow_calculation=False)
