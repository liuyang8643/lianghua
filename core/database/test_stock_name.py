"""stock_name 模块集成测试（需要网络访问 CNINFO）"""
import unittest
from datetime import date

from core.database.stock_name import (
    get_stock_name_at_date,
    is_st_at_date,
    _fetch_name_changes,
    _fetch_st_changes,
    clear_stock_name_cache,
)


class TestFetchNameChanges(unittest.TestCase):
    """测试 CNINFO p_stock2109 数据拉取"""

    def test_sh_stock_600186(self):
        records = _fetch_name_changes("600186")
        self.assertGreater(len(records), 0, "600186 应有更名记录")
        # 2019-04-29 莲花健康 → *ST莲花
        st_record = [r for r in records if "ST" in r["old_name"]]
        self.assertTrue(st_record, "应包含 *ST莲花 的更名记录")

    def test_sz_stock_000023(self):
        records = _fetch_name_changes("000023")
        self.assertGreater(len(records), 0, "000023 应有更名记录")

    def test_no_rename_stock(self):
        # 贵州茅台 600519 - 从未改名
        records = _fetch_name_changes("600519")
        self.assertEqual(len(records), 0, "贵州茅台不应有更名记录")


class TestFetchSTChanges(unittest.TestCase):
    """测试 CNINFO p_stock2117 数据拉取"""

    def test_sh_stock_600186(self):
        records = _fetch_st_changes("600186")
        self.assertGreater(len(records), 1, "600186 应有多条特别处理记录")
        events = [r["event"] for r in records]
        self.assertIn("戴帽披*", events, "应包含 戴帽披* 事件")

    def test_normal_stock_no_st(self):
        records = _fetch_st_changes("600519")
        # 贵州茅台只有一条"新股上市"记录
        st_events = [r for r in records if "ST" in r.get("status", "")]
        self.assertEqual(len(st_events), 0, "贵州茅台不应有 ST 记录")


class TestGetStockNameAtDate(unittest.TestCase):
    """测试历史日期股票名称查询"""

    @classmethod
    def setUpClass(cls):
        clear_stock_name_cache()

    def test_600186_before_st(self):
        name = get_stock_name_at_date("600186.SH", date(2019, 4, 1))
        self.assertIsNotNone(name)
        self.assertNotIn("ST", name, "2019-04-01 应还不是 ST")

    def test_600186_during_st(self):
        name = get_stock_name_at_date("600186.SH", date(2019, 5, 1))
        self.assertIsNotNone(name)
        self.assertIn("ST", name, "2019-05-01 应为 *ST莲花")

    def test_600186_after_st_removal(self):
        name = get_stock_name_at_date("600186.SH", date(2020, 5, 1))
        self.assertIsNotNone(name)
        self.assertNotIn("ST", name, "2020-05-01 应已摘帽")

    def test_no_rename_stock(self):
        name = get_stock_name_at_date("600519.SH", date(2023, 1, 1))
        self.assertIsNotNone(name)
        self.assertIn("茅台", name)


class TestIsSTAtDate(unittest.TestCase):
    """测试历史 ST 状态判断"""

    def test_600186_not_st(self):
        self.assertFalse(is_st_at_date("600186.SH", date(2018, 1, 1)))

    def test_600186_is_st(self):
        self.assertTrue(is_st_at_date("600186.SH", date(2019, 6, 1)))

    def test_600186_st_removed(self):
        self.assertFalse(is_st_at_date("600186.SH", date(2021, 1, 1)))

    def test_normal_stock(self):
        self.assertFalse(is_st_at_date("600519.SH", date(2023, 1, 1)))


if __name__ == "__main__":
    unittest.main()
