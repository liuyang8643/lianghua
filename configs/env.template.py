import os

# 日志文件存放路径
current_work_dir = os.path.dirname(__file__)
LOGGER_PATH = os.path.join(current_work_dir, 'logs')

# 国金 QMT 安装路径列表（使用第一个存在的目录）
QMT_ROOT_DIR = [
  'Z:\\gjqmt\\bin.x64',
  'D:\\gjqmt\\bin.x64',
]

# 交易账户配置
TRADE_ACCOUNT = '你的账户ID'

# 飞书机器人配置（回测可留空）
LARK_APP_ID = "你的飞书机器人ID"
LARK_APP_SECRET = "你的飞书机器人密钥"
LARK_RECEIVE_ID = "你的飞书chat_id"
