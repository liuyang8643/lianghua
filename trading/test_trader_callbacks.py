import trading.trader as trader_module


class _FakeXtTrader:
    instances = []

    def __init__(self, *args):
        self.registered = []
        self.__class__.instances.append(self)

    def set_relaxed_response_order_enabled(self, enabled):
        pass

    def register_callback(self, callback):
        self.registered.append(callback)

    def start(self):
        pass

    def connect(self):
        return 0

    def subscribe(self, account):
        return 0


def test_dry_run_trader_does_not_register_qmt_callback(monkeypatch):
    _FakeXtTrader.instances.clear()
    monkeypatch.setattr(trader_module, 'XtQuantTrader', _FakeXtTrader)
    monkeypatch.setattr(trader_module.xtdata, 'get_data_dir', lambda: 'C:/fake/userdata_mini/datadir')

    trader = trader_module.Trader('test', enable_callbacks=False, backend='native')

    assert trader.callback is None
    assert _FakeXtTrader.instances[-1].registered == []


def test_live_trader_registers_qmt_callback(monkeypatch):
    _FakeXtTrader.instances.clear()
    monkeypatch.setattr(trader_module, 'XtQuantTrader', _FakeXtTrader)
    monkeypatch.setattr(trader_module.xtdata, 'get_data_dir', lambda: 'C:/fake/userdata_mini/datadir')

    trader = trader_module.Trader('test', enable_callbacks=True, backend='native')

    assert trader.callback is not None
    assert _FakeXtTrader.instances[-1].registered == [trader.callback]


def test_bridge_mode_does_not_require_native_market_data_connection(monkeypatch):
    import trading.bridge_client as bridge_module

    def unavailable():
        raise RuntimeError('Native xtdata unavailable')

    monkeypatch.setattr(trader_module.xtdata, 'get_data_dir', unavailable)
    monkeypatch.setattr(bridge_module, 'BridgeClient', _FakeXtTrader)
    trader = trader_module.Trader('test', enable_callbacks=False, enable_orders=False,
                                  backend='bridge')
    assert trader.backend == 'bridge'
