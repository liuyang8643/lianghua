from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_trading_uses_canonical_trade_and_env_order_plan_chain():
    main_source = (ROOT / 'trade' / 'main.py').read_text(encoding='utf-8')
    post_close_source = (ROOT / 'trade' / 'post_close.py').read_text(encoding='utf-8')
    runtime_source = (ROOT / 'trade' / 'runtime.py').read_text(encoding='utf-8')
    executor_source = (ROOT / 'trade' / 'executor.py').read_text(encoding='utf-8')
    journal_source = (ROOT / 'trade' / 'journal.py').read_text(encoding='utf-8')

    assert 'from trade.runtime import' in main_source
    assert 'from trade.executor import' in main_source
    assert 'from trade.journal import' in main_source
    assert 'LiveDecisionRunner' in main_source
    assert 'broker_executor.execute(decision.order_plan)' in main_source
    assert 'from env.contracts import' in executor_source
    assert 'OrderPlan' in executor_source
    assert 'policy.predict(observation, deterministic=True)' in runtime_source
    assert 'journal.replay(trade_date)' in post_close_source
    assert '.predict(' not in post_close_source
    assert '.predict(' not in journal_source

    forbidden = (
        'build_strategy_day',
        'build_rebalance_day',
        'core.timing',
        'compute_position_multiplier',
        'position_multiplier',
        'holding_period',
        'target_cash',
        'target_positions',
        '_compute_factor_scores',
    )
    live_sources = main_source + post_close_source + runtime_source + executor_source
    for token in forbidden:
        assert token not in live_sources
