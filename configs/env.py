import os

# 日志文件存放路径
current_work_dir = os.path.dirname(__file__)
LOGGER_PATH = os.path.join(current_work_dir, 'logs')

# QMT 安装路径列表（使用第一个存在的目录）
QMT_ROOT_DIR = [
  'D:\\申万宏源策略量化交易终端\\bin.x64'
]

# 交易账户配置
TRADE_ACCOUNT = '824300045987'

# 飞书机器人配置（回测可留空）
LARK_APP_ID = "cli_a92a96c39d785cd5"
LARK_APP_SECRET = "2agI94VvgyhP0FKBeowcqhUggH7A7Rm6"
LARK_RECEIVE_ID = "oc_73029acae6daaf5f2fc4e4f841918c46"
