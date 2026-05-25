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
    self._configured = bool(app_id and app_secret)
    if self._configured:
      self.client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
    else:
      self.client = None

  @staticmethod
  def log_res(res: lark.BaseResponse):
    if not res.success():
      lark.logger.error(f"发送消息失败：[{res.code}]{res.msg}")
    return res

  def _skip(self, method):
    if not self._configured:
      return lark.BaseResponse()

  def send_msg(self, msg: str):
    if not self._configured:
      return
    create_message_req = (
      CreateMessageRequest
      .builder()
      .receive_id_type('chat_id')
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
    if not self._configured:
      return
    create_message_req = (
      CreateMessageRequest
      .builder()
      .receive_id_type('chat_id')
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

  def send_file(self, file_path: str, file_name: str = None):
    if not self._configured:
      return
    import os, io
    file_name = file_name or os.path.basename(file_path)
    with open(file_path, 'rb') as f:
      file_content = f.read()
    upload_req = (
      CreateFileRequest.builder()
      .request_body(
        CreateFileRequestBody.builder()
        .file_name(file_name)
        .file_type("stream")
        .file(io.BytesIO(file_content))
        .build()
      ).build()
    )
    upload_resp = self.client.im.v1.file.create(upload_req)
    if not upload_resp.success():
      lark.logger.error(f"文件上传失败: [{upload_resp.code}]{upload_resp.msg}")
      return upload_resp
    file_key = upload_resp.data.file_key
    msg_req = (
      CreateMessageRequest.builder()
      .receive_id_type('chat_id')
      .request_body(
        CreateMessageRequestBody.builder()
        .receive_id(LARK_RECEIVE_ID)
        .msg_type("file")
        .content(lark.JSON.marshal({"file_key": file_key}))
        .build()
      ).build()
    )
    res = self.client.im.v1.message.create(msg_req)
    return self.log_res(res)

  def send_notification_card(
      self,
      content: str,
      level: LarkMsgLevel = LarkMsgLevel.Success,
      title: str = None,
      sub_title: str = None,

  ):
    lines = []
    if title:
      lines.append(f"【{title}】")
    if sub_title:
      lines.append(sub_title)
    if content:
      lines.append(content)
    return self.send_msg("\n".join(lines))

lark_sender = LarkSender(LARK_APP_ID, LARK_APP_SECRET)

if __name__ == "__main__":
  lark_sender.send_notification_card(
    level=LarkMsgLevel.Danger,
    title="这是标题",
    content="这是内容",
  )
