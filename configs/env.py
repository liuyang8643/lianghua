import os

# 日志文件存放路径
current_work_dir = os.path.dirname(__file__)
LOGGER_PATH = os.path.join(current_work_dir, 'logs')

# QMT 安装路径列表（使用第一个存在的目录）
QMT_ROOT_DIR = [
  'D:\\申万宏源策略量化交易终端\\bin.x64'
]

# 交易账户配置。生产凭据只从进程环境注入，不进入源码或模型包。
TRADE_ACCOUNT = os.environ.get("WBR_TRADE_ACCOUNT", "")

# 飞书机器人配置（回测可留空）
LARK_APP_ID = os.environ.get("WBR_LARK_APP_ID", "")
LARK_APP_SECRET = os.environ.get("WBR_LARK_APP_SECRET", "")
LARK_RECEIVE_ID = os.environ.get("WBR_LARK_RECEIVE_ID", "")
