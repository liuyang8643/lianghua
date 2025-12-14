from datetime import date, datetime
from core.factors import MACD
from core.factors.helpers import FactorCtx

# 测试 date 对象
test_date = date(2024, 12, 10)
print(f"测试日期: {test_date}, 类型: {type(test_date)}")

# 创建 FactorCtx
try:
    ctx = FactorCtx("600051.SH", test_date)
    print(f"FactorCtx.base_time: {ctx.base_time}, 类型: {type(ctx.base_time)}")

    # 计算MACD因子
    macd_factor = MACD()
    result = macd_factor.calc(ctx)
    print(f"MACD结果: {result}")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
