import lark_oapi as lark
from enum import Enum
from lark_oapi.api.im.v1 import *

from configs import LARK_APP_ID, LARK_APP_SECRET, LARK_RECEIVE_ID

class LarkMsgLevel(Enum):
  Success = 'green'
  Info = 'wathet'
  Warning = 'orange'
  Danger = 'red'

class LarkSender:
  def __init__(self, app_id: str, app_secret: str):
    self.client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()

  @staticmethod
  def log_res(res: lark.BaseResponse):
    if not res.success():
      lark.logger.error(f"发送消息失败：[{res.code}]{res.msg}")
    return res

  def send_msg(self, msg: str):
    create_message_req = (
      CreateMessageRequest
      .builder()
      .receive_id_type('email')
      .request_body(
        CreateMessageRequestBody
        .builder()
        .receive_id(LARK_RECEIVE_ID)
        .msg_type("text")
        .content(lark.JSON.marshal({"text": msg}))
        .build()
      ).build()
    )
    res = self.client.im.v1.message.create(create_message_req)
    return self.log_res(res)

  def send_card(self, card_data: dict):
    create_message_req = (
      CreateMessageRequest
      .builder()
      .receive_id_type('email')
      .request_body(
        CreateMessageRequestBody
        .builder()
        .receive_id(LARK_RECEIVE_ID)
        .msg_type("interactive")
        .content(
          lark.JSON.marshal(
            {
              'type': 'template',
              'data': card_data
            }
          )
        )
        .build()
      ).build()
    )
    res = self.client.im.v1.message.create(create_message_req)
    return self.log_res(res)

  def send_notification_card(
      self,
      content: str,
      level: LarkMsgLevel = LarkMsgLevel.Success,
      title: str = None,
      sub_title: str = None,

  ):
    return self.send_card(
      {
        "template_id": 'AAqS6QpkHvP8I',
        "template_variable": {
          "title": title if title else '',
          "sub_title": sub_title if sub_title else '',
          "msg_color": level.value,
          "content": content,
        }
      }
    )

lark_sender = LarkSender(LARK_APP_ID, LARK_APP_SECRET)

if __name__ == "__main__":
  lark_sender.send_notification_card(
    level=LarkMsgLevel.Danger,
    title="这是标题",
    content="这是内容",
  )
