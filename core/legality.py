"""买卖合法性闸门 —— 回测与实盘的唯一实现（CLAUDE.md §2.2 对齐红线）。

本模块是回测准不准的核心：选股之后、成交之前的最后一道闸，判定每只候选股在 T 日
能否按"收盘价成交"（尾盘 15:00-15:30 收盘集合竞价固定价）。涨跌停规则随板块（主板/
创业板/科创板/北交所）与时间段（注册制改革、ST 规则调整）变化，逻辑复杂，故独立成
`LegalityChecker` 类集中维护。

红线：选股/合法性 **只允许使用 close[T] 与 preClose[T]**（前收）。T 日 open/high/low/
volume/amount 一律不参与判定。

收盘封板即无法成交：`close[T] ≥ 涨停价` 视为收盘封涨停（禁买），`close[T] ≤ 跌停价`
视为收盘封跌停（禁卖）。IPO 首日盘中涨停幅为发行价×1.44，由通用涨停判定（ratios=0.44）
直接覆盖，无需专项 HLC 逻辑。

取整偏严：涨停价向下取整、跌停价向上取整，使"触及涨跌停"更易成立、买卖更易被拒。
偏严方向回测可与实盘对齐，宽松则无法对齐。
"""
import numpy as np
from datetime import date

_EPS = 0.001

# —— 制度/板块切换基准日（用 signal_date 判定）——
_IPO_44_START = date(2014, 1, 1)     # 新股首日"开盘≤+20%、盘中≤+44%"特殊机制生效
_KCB_OPEN = date(2019, 7, 22)        # 科创板开市：前5日不设限、日常±20%
_CYB_REG = date(2020, 8, 24)         # 创业板注册制：前5日不设限、日常±20%、ST±20%
_MB_REG = date(2023, 4, 10)          # 主板全面注册制：前5日不设限
_MB_ST_10_START = date(2026, 7, 6)   # 主板风险警示股涨跌幅 5%→10%

# 临退禁买窗口（交易日）：覆盖退市整理期(主板/创业板15日、科创板30日)+停牌缓冲。
# 宁严勿漏——临退期"首日不设涨跌幅"无法对齐实盘撮合，且有归零风险，一律禁买。
_DELIST_BUY_BAN_DAYS = 35

# 板块编码
BOARD_MAIN = 0   # 主板（沪深，含原中小板 002/003）
BOARD_CYB = 1    # 创业板 300/301
BOARD_KCB = 2    # 科创板 688
BOARD_BJ = 3     # 北交所 83/87/43/92


def _floor_2(values):
    """向下取整到分。用于涨停价：价格【算低】→ 更易判触顶
    → 更易禁买（偏严）。偏严方向回测可与实盘对齐，宽松则无法对齐。"""
    return np.floor(values * 100.0 + 1e-9) / 100.0


def _ceil_2(values):
    """向上取整到分。用于跌停价：价格【算高】→ 更易判触底 → 更易禁卖（偏严）。"""
    return np.ceil(values * 100.0 - 1e-9) / 100.0


class LegalityChecker:
    """A 股买卖合法性闸门。回测与实盘共用唯一实现，禁止出现第二份。

    用法：
        checker = LegalityChecker(data, stock_indices, list_dates_map, delist_dates_map)
        mask, reasons = checker.check(candidates_idx, trade_idx, signal_date, is_buy=True)

        其中 candidates_idx 是候选股在 NPZ 列空间的索引数组；mask[i] 为 True 表示第 i 只
        候选股 T 日收盘可成交。一次性预计算（板块/上市日/退市日）在 __init__，每个调仓日
        只做向量化布尔运算，无逐股票循环。
    """

    def __init__(self, data, stock_indices, list_dates_map=None, delist_dates_map=None,
                 limit_up_protection=False):
        """
        Args:
            data: load_runtime_npz 返回的 dict（含 close/preClose/st_mask/issue_price 等）
            stock_indices: {code: col_idx}
            list_dates_map: {code: list_date} 可选；回测从 K 线首日推断
            delist_dates_map: {code: delist_date} 可选；回测从 db.delist 传入，用于临退禁买。
                              实盘不传（由 allow_buy 名单剔除退市股），delist_tidx 全 -1 不生效。
            limit_up_protection: 一字涨停保护——卖出侧也过滤涨停股，避免卖掉封板标的。

        涨跌停判定一律用「原始 close + 官方 preClose（除权除息参考价）」：preClose 本身已吸收
        分红送转配股，除权日 close/preClose 天然不会假跳空——无需复权价、研究/对账口径完全一致。
        """
        self.close_all = data['close']
        self.preclose_all = data['preClose']
        self.st_all = data['st_mask']
        self.issue_price_all = data['issue_price']
        self.limit_up_protection = limit_up_protection
        (self.board_type, self.base_ratio,
         self.list_tidx, self.delist_tidx) = self._precompute(
            data, stock_indices, list_dates_map, delist_dates_map)

    # ---------- 一次性预计算（非热路径）----------
    @staticmethod
    def _precompute(data, stock_indices, list_dates_map=None, delist_dates_map=None):
        """预计算 board_type / base_ratio / list_tidx / delist_tidx（整个回测只跑一次）。

        Returns:
            board_type (int8): 0主板 / 1创业板 / 2科创板 / 3北交所
            base_ratio (float64): 注册制后日常涨跌幅（主板10% / 双创20% / 北交所30%）
            list_tidx (int32): 上市日对应交易日索引，-1=无
            delist_tidx (int32): 退市(摘牌/暂停)日对应交易日索引，-1=无
        """
        codes = np.asarray([str(s) for s in data['stock_codes']], dtype='U12')
        n = len(codes)
        # 板块/基础涨跌幅：按代码前缀向量化判定
        bt = np.zeros(n, dtype=np.int8)
        br = np.full(n, 0.10, dtype=np.float64)
        is_cyb = np.char.startswith(codes, '300') | np.char.startswith(codes, '301')
        is_kcb = np.char.startswith(codes, '688')
        is_bj = (np.char.startswith(codes, '83') | np.char.startswith(codes, '87')
                 | np.char.startswith(codes, '43') | np.char.startswith(codes, '92'))
        bt[is_cyb] = BOARD_CYB; br[is_cyb] = 0.20
        bt[is_kcb] = BOARD_KCB; br[is_kcb] = 0.20
        bt[is_bj] = BOARD_BJ; br[is_bj] = 0.30

        tdp = [d.astype('datetime64[D]').item() for d in data['trade_dates']]
        d2t = {d: i for i, d in enumerate(tdp)}

        def _date_to_tidx(d):
            """日期→交易日索引；非交易日取其后第一个交易日。"""
            if d in d2t:
                return d2t[d]
            for dd in tdp:
                if dd >= d:
                    return d2t[dd]
            return None

        def _build_tidx(dates_map):
            arr = np.full(n, -1, dtype=np.int32)
            if dates_map:
                for code, dt in dates_map.items():
                    if code not in stock_indices:
                        continue
                    si = stock_indices[code]
                    ti = _date_to_tidx(dt)
                    if ti is not None:
                        arr[si] = ti
            return arr

        return bt, br, _build_tidx(list_dates_map), _build_tidx(delist_dates_map)

    # ---------- 每个调仓日的合法性判定（热路径，全向量化）----------
    def check(self, candidates_idx, trade_idx, signal_date, is_buy):
        """向量化涨跌停 / IPO 首日 / 临退检查（T 日收盘成交契约）。

        Args:
            candidates_idx: 候选股在 NPZ 列空间的索引（list 或 ndarray）
            trade_idx: T 日在 NPZ 的行索引
            signal_date: T 日日期（datetime.date），用于板块/制度时段判定
            is_buy: True=买入(查涨停+临退)，False=卖出(查跌停)

        Returns:
            (mask, reasons): mask[i]=True 表示可成交；reasons 含停牌计数

        数据使用：close[T] / preClose[T]（官方前收）/ st_mask[T] / 发行价。

        已知局限：重新上市/恢复上市/增发上市首日同属"不设涨跌幅"，runtime 暂无事件日期
        字段，无法单独识别其非跳空形态；收盘封板形态已被涨停判定一并拦下。
        """
        idx = np.asarray(candidates_idx, dtype=np.intp)
        n = len(idx)
        if n == 0:
            return np.array([], dtype=bool), {}

        closes = self.close_all[trade_idx, idx].astype(np.float64)
        valid_close = ~np.isnan(closes) & (closes > 0)
        if not np.any(valid_close):
            return np.zeros(n, dtype=bool), {'suspended': n}

        # 前收 = preClose[T]（除权除息参考价），与原始 close 同口径判涨跌停
        # NaN preClose（非标事件如转配股）→ 该股当日不可交易
        precloses = self.preclose_all[trade_idx, idx].astype(np.float64)
        valid_preclose = ~np.isnan(precloses) & (precloses > 0)

        boards = self.board_type[idx]
        lti = self.list_tidx[idx]

        # ===== 1. 日常涨跌幅比例（非首日、非 ST）=====
        ratios = self.base_ratio[idx].astype(np.float64).copy()
        # 创业板注册制前日常 ±10%（base_ratio 存的是注册制后的 0.20）
        ratios[(boards == BOARD_CYB) & (signal_date < _CYB_REG)] = 0.10

        # ===== 2. ST/风险警示按板块覆盖 =====
        #   主板 ±5%(2026-07-06 起 ±10%) / 创业板 注册制前±5%·注册制后±20% / 科创板 ±20% / 北交所 ±30%
        st_arr = self.st_all[trade_idx, idx]
        if np.any(st_arr):
            cyb_st = 0.05 if signal_date < _CYB_REG else 0.20
            main_st = 0.05 if signal_date < _MB_ST_10_START else 0.10
            st_ratio = np.select(
                [boards == BOARD_CYB, boards == BOARD_KCB, boards == BOARD_BJ],
                [cyb_st, 0.20, 0.30], default=main_st)
            ratios = np.where(st_arr, st_ratio, ratios)

        # ===== 3. 上市初期判定（全向量化，无逐股票循环）=====
        listed = (lti >= 0) & (trade_idx >= lti)
        ds = trade_idx - lti  # 上市后第几个交易日（0=首日），仅 listed 有效
        # 3a. 不设涨跌幅的"可买新股初期"：注册制前5日 / 北交所首日 / 2014 前首日
        exempt = (boards == BOARD_BJ) & listed & (ds == 0)               # 北交所：上市首日
        if signal_date >= _KCB_OPEN:
            exempt |= (boards == BOARD_KCB) & listed & (ds <= 4)         # 科创板：前5日
        if signal_date >= _CYB_REG:
            exempt |= (boards == BOARD_CYB) & listed & (ds <= 4)         # 创业板注册制：前5日
        elif signal_date < _IPO_44_START:
            exempt |= (boards == BOARD_CYB) & listed & (ds == 0)         # 2014 前：首日
        if signal_date >= _MB_REG:
            exempt |= (boards == BOARD_MAIN) & listed & (ds <= 4)        # 主板全面注册制：前5日
        elif signal_date < _IPO_44_START:
            exempt |= (boards == BOARD_MAIN) & listed & (ds == 0)        # 2014 前：首日
        first_list_day = listed & (ds == 0)
        # 3b. 老规则 IPO 首日（主板2014~2023-04-09 / 创业板2014~2020-08-23）：盘中涨停=发行价×1.44
        #     收盘价触及 1.44 即视为封板禁买（由下方 ratios=0.44 的通用涨停判定覆盖）。
        is_ipo_first = listed & (trade_idx == lti) & (signal_date >= _IPO_44_START) & ~exempt

        # 所有上市首日涨跌幅基准均使用发行价；豁免期仍不参与涨跌停判定。
        if np.any(first_list_day):
            ips = self.issue_price_all[idx].astype(np.float64)
            vip = first_list_day & ~np.isnan(ips) & (ips > 0)
            precloses[vip] = ips[vip]; valid_preclose[vip] = True

        ratios = np.where(is_ipo_first, 0.44, ratios)   # is_ipo_first 已排除 exempt
        has_limit = valid_preclose & ~exempt            # exempt(不设涨跌幅) 不参与涨跌停判定

        if is_buy:
            tradable = self._check_buy(idx, trade_idx, closes, precloses, ratios,
                                       valid_close, has_limit)
        else:
            down_limits = np.where(has_limit, _ceil_2(precloses * (1.0 - ratios)), np.nan)
            limit_down = valid_close & has_limit & (closes <= down_limits + _EPS)
            tradable = valid_close & ~limit_down
            if self.limit_up_protection:
                up_limits = np.where(has_limit, _floor_2(precloses * (1.0 + ratios)), np.nan)
                limit_up = valid_close & has_limit & (closes >= up_limits - _EPS)
                tradable = tradable & ~limit_up

        # preClose NaN（转配股等非标事件）→ 非豁免期股票无法判定涨跌停，当日跳过
        tradable = tradable & (valid_preclose | exempt)

        suspended = int(np.sum(~valid_close))
        reasons = {'suspended': suspended} if suspended > 0 else {}
        return tradable, reasons

    # ---------- 买入侧：收盘涨停 + 临退 ----------
    def _check_buy(self, idx, trade_idx, closes, precloses, ratios,
                   valid_close, has_limit):
        # ===== 4. 收盘涨停禁买（涨停价向下取整，偏严）=====
        #   收盘封涨停即买不进。IPO 首日盘中涨停=发行价×1.44 由 ratios=0.44 覆盖，无需专项。
        up_limits = np.where(has_limit, _floor_2(precloses * (1.0 + ratios)), np.nan)
        limit_up = valid_close & has_limit & (closes >= up_limits - _EPS)
        tradable = valid_close & ~limit_up

        # ===== 5. 临退禁买（退市整理期首日不设涨跌幅 + 归零风险，宁严勿买）=====
        if self.delist_tidx is not None:
            dlt = self.delist_tidx[idx]
            delist_zone = (dlt >= 0) & (trade_idx >= dlt - _DELIST_BUY_BAN_DAYS)
            tradable = tradable & ~delist_zone

        return tradable
