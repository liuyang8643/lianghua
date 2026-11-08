from pathlib import Path
from typing import Optional

import pandas as pd

import logging

data_logger = logging.getLogger(__name__)
from data.db.type import StockDetail
from data.db.delist import get_delist_stock_info

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_stock_list() -> pd.DataFrame:
    path = _DATA_DIR / "stock_list" / "stock_list.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _build_detail_from_local(stock_code: str) -> Optional[StockDetail]:
    """从本地 parquet 构建 StockDetail，无网络依赖。"""
    bare_code = stock_code.split('.')[0]

    if stock_code.endswith('.SH'):
        exchange = 'SSE'
    elif stock_code.endswith('.SZ'):
        exchange = 'SZE'
    elif stock_code.endswith('.BJ'):
        exchange = 'BSE'
    else:
        return None

    df_list = _load_stock_list()
    if not df_list.empty and stock_code not in df_list['stock_code'].values:
        return None

    open_date = '19900101'
    expire_date = '99999999'
    instrument_status = 3
    delist_info = get_delist_stock_info()
    if stock_code in delist_info:
        info = delist_info[stock_code]
        open_date = info.list_date.strftime('%Y%m%d')
        expire_date = info.delist_date.strftime('%Y%m%d')
        instrument_status = 1
    else:
        from data.db.stock_list import _get_stock_date_range
        date_range = _get_stock_date_range(stock_code)
        if date_range and date_range[0]:
            open_date = date_range[0].strftime('%Y%m%d')

    from data.db.stock_name import get_current_stock_name
    name = get_current_stock_name(stock_code) or ''

    return {
        'ExchangeID': exchange,
        'InstrumentID': bare_code,
        'InstrumentName': name,
        'ProductID': '',
        'ProductName': '',
        'ProductType': -1,
        'ExchangeCode': stock_code,
        'UniCode': stock_code,
        'CreateDate': int(open_date),
        'OpenDate': open_date,
        'ExpireDate': expire_date,
        'PreClose': 0.0,
        'SettlementPrice': 0.0,
        'UpStopPrice': 0.0,
        'DownStopPrice': 0.0,
        'FloatVolume': 0.0,
        'TotalVolume': 0.0,
        'PriceTick': 0.01,
        'VolumeMultiple': 1,
        'MainContract': 0,
        'LastVolume': 0,
        'InstrumentStatus': instrument_status,
    }


def get_stock_detail(stock_code: str) -> Optional[StockDetail]:
    """获取股票详情（从本地 parquet 构建）。"""
    try:
        return _build_detail_from_local(stock_code)
    except Exception as e:
        data_logger.error(f'获取详情失败: {stock_code}, {e}')
    return None
