import unittest
from datetime import date
from unittest.mock import patch

from utils.stock.info import (
  _round_limit_price,
  get_limit_band_from_ratio,
  is_bse_stock,
  resolve_limit_regime,
)


def _detail(name: str = '测试股票', open_date: str = '20200101') -> dict:
  return {
    'InstrumentName': name,
    'OpenDate': open_date,
    'CreateDate': open_date,
  }


class TestInfo(unittest.TestCase):
  def test_is_bse_stock_supports_legacy_and_new_prefixes(self):
    self.assertTrue(is_bse_stock('430047.BJ'))
    self.assertTrue(is_bse_stock('830799.BJ'))
    self.assertTrue(is_bse_stock('870436.BJ'))
    self.assertTrue(is_bse_stock('920008.BJ'))
    self.assertFalse(is_bse_stock('600000.SH'))

  def test_round_limit_price_uses_half_up_for_down_limit(self):
    self.assertEqual(_round_limit_price(10.05, 0.10, True), 11.06)
    self.assertEqual(_round_limit_price(10.05, 0.10, False), 9.05)
    self.assertEqual(_round_limit_price(9.95, 0.10, False), 8.96)

  def test_get_limit_band_from_ratio_uses_correct_down_limit_rounding(self):
    bar = {
      'preClose': 10.05,
    }

    with patch('utils.stock.info._get_historical_st_checker', return_value=lambda *_: False):
      up_limit, down_limit, regime = get_limit_band_from_ratio(
        '600000.SH',
        date(2024, 1, 10),
        bar,
        detail=_detail(),
      )

    self.assertEqual(up_limit, 11.06)
    self.assertEqual(down_limit, 9.05)
    self.assertEqual(regime['name'], 'main_board')

  def test_resolve_limit_regime_uses_historical_st_for_main_board(self):
    with patch('utils.stock.info._get_historical_st_checker', return_value=lambda *_: True):
      regime = resolve_limit_regime(
        '600186.SH',
        date(2019, 5, 1),
        detail=_detail(name='莲花健康'),
      )

    self.assertEqual(regime['name'], 'st')
    self.assertEqual(regime['ratio'], 0.05)
    self.assertTrue(regime['is_st'])

  def test_resolve_limit_regime_keeps_cyb_st_at_board_limit(self):
    with patch('utils.stock.info._get_historical_st_checker', return_value=lambda *_: True):
      regime = resolve_limit_regime(
        '300506.SZ',
        date(2024, 1, 10),
        detail=_detail(name='名家汇'),
      )

    self.assertEqual(regime['name'], 'cyb')
    self.assertEqual(regime['ratio'], 0.20)
    self.assertTrue(regime['is_st'])

  def test_resolve_limit_regime_keeps_kcb_st_at_board_limit(self):
    with patch('utils.stock.info._get_historical_st_checker', return_value=lambda *_: True):
      regime = resolve_limit_regime(
        '688001.SH',
        date(2024, 1, 10),
        detail=_detail(name='华兴源创'),
      )

    self.assertEqual(regime['name'], 'kcb')
    self.assertEqual(regime['ratio'], 0.20)
    self.assertTrue(regime['is_st'])

  def test_resolve_limit_regime_recognizes_920_as_bse(self):
    with patch('utils.stock.info._get_historical_st_checker', return_value=lambda *_: False):
      regime = resolve_limit_regime(
        '920008.BJ',
        date(2024, 8, 26),
        detail=_detail(name='成电光信'),
      )

    self.assertEqual(regime['name'], 'bse')
    self.assertEqual(regime['ratio'], 0.30)
    self.assertFalse(regime['is_st'])

  def test_resolve_st_status_falls_back_to_detail_when_history_lookup_fails(self):
    with patch('utils.stock.info._get_historical_st_checker', side_effect=RuntimeError('boom')):
      regime = resolve_limit_regime(
        '600000.SH',
        date(2024, 1, 10),
        detail=_detail(name='ST浦发'),
      )

    self.assertEqual(regime['name'], 'st')
    self.assertEqual(regime['ratio'], 0.05)
    self.assertTrue(regime['is_st'])


if __name__ == '__main__':
  unittest.main()
