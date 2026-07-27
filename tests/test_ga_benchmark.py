from datetime import date, datetime

from testback.run_ga import _compute_benchmark_metric
from utils.stock.time import get_trading_date_span


def _trading_datetimes(start: date, end: date) -> list[datetime]:
    return [
        datetime.combine(day, datetime.min.time())
        for day in get_trading_date_span(start, end)
    ]


def test_csi500_benchmark_covers_the_full_training_period():
    dates = _trading_datetimes(date(2010, 1, 1), date(2018, 12, 31))
    metric = _compute_benchmark_metric('sh000905', dates)

    assert metric is not None
    assert metric['available_days'] == metric['requested_days']
    assert metric['start'] == '2010-01-04'


def test_csi1000_benchmark_marks_partial_training_coverage():
    dates = _trading_datetimes(date(2010, 1, 1), date(2018, 12, 31))
    metric = _compute_benchmark_metric('sh000852', dates)

    assert metric is not None
    assert metric['available_days'] < metric['requested_days']
    assert metric['start'] == '2014-10-17'
