"""飞书通知发送器。

设计：
1. **统一 v2 schema**：所有卡片走 JSON Schema 2.0，header.template 用颜色枚举。
2. **自动 audit 留存**：每次飞书发送（无论成功失败）都按日落地到
   `data/live_trades/lark_audit/{YYYYMMDD}.jsonl`，含完整 payload + response。
3. **三层接口**：
     - `send_msg(text)`           纯文本
     - `send_notification_card`   简易「标题+副标题+正文」卡片
     - `send_card(card_dict)`     直传 v2 schema dict（最大自由度）
     - `send_table_card(...)`     v2 schema 原生 table 组件快捷封装
     - `send_file(path)`          上传文件附件
"""
import io
import json
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

from configs import LARK_APP_ID, LARK_APP_SECRET, LARK_RECEIVE_ID


# ─────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────
_AUDIT_DIR = Path(__file__).resolve().parents[2] / "data" / "live_trades" / "lark_audit"


def _audit(record: dict):
  """落地一条飞书发送记录到当日 jsonl。"""
  try:
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    record.setdefault('ts', datetime.now().isoformat(timespec='milliseconds'))
    path = _AUDIT_DIR / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(path, 'a', encoding='utf-8') as f:
      f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
  except Exception as e:
    print(f"[LarkAudit] 写入失败: {e}")


class LarkMsgLevel(Enum):
  Success = 'green'
  Info = 'wathet'
  Warning = 'orange'
  Danger = 'red'


# ─────────────────────────────────────────────
# v2 Schema 通用组件构造
# ─────────────────────────────────────────────

def make_v2_card(*, title: str, level: LarkMsgLevel = LarkMsgLevel.Info,
                 subtitle: str | None = None,
                 elements: list[dict] | None = None) -> dict:
  """生成一个 v2 schema 卡片骨架。"""
  header: dict[str, Any] = {
      'title': {'tag': 'plain_text', 'content': title or ''},
      'template': level.value,
  }
  if subtitle:
    header['subtitle'] = {'tag': 'plain_text', 'content': subtitle}
  return {
      'schema': '2.0',
      'config': {'update_multi': True},
      'header': header,
      'body': {'elements': list(elements or [])},
  }


def md_div(content: str) -> dict:
  """lark_md div 块。"""
  return {'tag': 'div', 'text': {'tag': 'lark_md', 'content': content}}


def make_v2_table(*, columns: list[dict], rows: list[dict],
                  element_id: str = 'tbl', page_size: int = 10,
                  freeze_first_column: bool = True) -> dict:
  """生成一个 v2 schema table 组件（统一 lark_md 数据类型，避免 horizontal_align 兼容问题）。"""
  norm_cols = []
  for c in columns:
    norm_cols.append({
        'name': c['name'],
        'display_name': c.get('display_name', c['name']),
        'data_type': c.get('data_type', 'lark_md'),
        'horizontal_align': c.get('horizontal_align', 'left'),
    })
  return {
      'tag': 'table',
      'element_id': element_id,
      'page_size': page_size,
      'row_height': 'low',
      'freeze_first_column': freeze_first_column,
      'header_style': {'text_align': 'center', 'bold': True,
                       'background_style': 'grey', 'text_color': 'grey'},
      'columns': norm_cols,
      'rows': rows,
  }


# ─────────────────────────────────────────────
# Sender
# ─────────────────────────────────────────────
class LarkSender:
  def __init__(self, app_id: str, app_secret: str):
    self._configured = bool(app_id and app_secret)
    if self._configured:
      self.client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
    else:
      self.client = None

  @staticmethod
  def _summarize_res(res):
    """提取 BaseResponse 关键字段。"""
    if res is None:
      return {'ok': False, 'code': None, 'msg': 'no response'}
    return {
        'ok': bool(res.success()),
        'code': getattr(res, 'code', None),
        'msg': getattr(res, 'msg', None),
    }

  def _log_and_audit(self, res, audit_record: dict):
    info = self._summarize_res(res)
    if not info['ok']:
      lark.logger.error(f"发送消息失败：[{info['code']}]{info['msg']}")
    audit_record['ok'] = info['ok']
    audit_record['response'] = {'code': info['code'], 'msg': info['msg']}
    _audit(audit_record)
    return res

  # ── 纯文本 ─────────────────────────────────────
  def send_msg(self, msg: str):
    if not self._configured:
      _audit({'method': 'send_msg', 'content': msg, 'skipped': 'not_configured'})
      return
    try:
      req = (CreateMessageRequest.builder()
             .receive_id_type('chat_id')
             .request_body(CreateMessageRequestBody.builder()
                           .receive_id(LARK_RECEIVE_ID).msg_type("text")
                           .content(lark.JSON.marshal({"text": msg})).build())
             .build())
      res = self.client.im.v1.message.create(req)
      return self._log_and_audit(res, {'method': 'send_msg', 'content': msg})
    except Exception as e:
      _audit({'method': 'send_msg', 'content': msg, 'ok': False, 'error': repr(e)})
      lark.logger.error(f"send_msg 异常: {e}")

  # ── v2 schema 卡片（任意 dict） ─────────────────
  def send_card(self, card_data: dict) -> str | None:
    """发送任意 v2 schema card。

    Returns:
        发送成功时返回 `message_id`（可用于后续 update_card 更新），失败返回 None。
    """
    if not self._configured:
      _audit({'method': 'send_card', 'card': card_data, 'skipped': 'not_configured'})
      return None
    try:
      req = (CreateMessageRequest.builder()
             .receive_id_type('chat_id')
             .request_body(CreateMessageRequestBody.builder()
                           .receive_id(LARK_RECEIVE_ID).msg_type("interactive")
                           .content(lark.JSON.marshal(card_data)).build())
             .build())
      res = self.client.im.v1.message.create(req)
      message_id = None
      if res and res.success() and res.data:
        message_id = getattr(res.data, 'message_id', None)
      self._log_and_audit(
          res,
          {'method': 'send_card',
           'title': card_data.get('header', {}).get('title', {}).get('content'),
           'template': card_data.get('header', {}).get('template'),
           'message_id': message_id,
           'card': card_data})
      return message_id
    except Exception as e:
      _audit({'method': 'send_card', 'card': card_data, 'ok': False, 'error': repr(e)})
      lark.logger.error(f"send_card 异常: {e}")
      return None

  # ── 更新已发送卡片 ─────────────────────────────
  def update_card(self, message_id: str, card_data: dict) -> bool:
    """通过 PATCH /im/v1/messages/:message_id 更新已发送的交互卡片。

    Args:
        message_id: send_card 返回的 message_id
        card_data: 完整 v2 schema 卡片 dict（替换式更新）

    Returns: True 成功 / False 失败

    注意：必须使用 PATCH（飞书"更新应用发送的消息卡片"接口），
          PUT 接口的 msg_type 不支持 interactive。
    """
    if not self._configured:
      _audit({'method': 'update_card', 'message_id': message_id,
              'card': card_data, 'skipped': 'not_configured'})
      return False
    if not message_id:
      return False
    try:
      req = (PatchMessageRequest.builder()
             .message_id(message_id)
             .request_body(PatchMessageRequestBody.builder()
                           .content(lark.JSON.marshal(card_data)).build())
             .build())
      res = self.client.im.v1.message.patch(req)
      info = self._summarize_res(res)
      if not info['ok']:
        lark.logger.error(f"update_card 失败: [{info['code']}]{info['msg']}")
      _audit({'method': 'update_card',
              'title': card_data.get('header', {}).get('title', {}).get('content'),
              'template': card_data.get('header', {}).get('template'),
              'message_id': message_id,
              'card': card_data,
              'ok': info['ok'],
              'response': {'code': info['code'], 'msg': info['msg']}})
      return info['ok']
    except Exception as e:
      _audit({'method': 'update_card', 'message_id': message_id,
              'card': card_data, 'ok': False, 'error': repr(e)})
      lark.logger.error(f"update_card 异常: {e}")
      return False

  # ── 文件上传 ───────────────────────────────────
  def send_file(self, file_path: str, file_name: str = None):
    if not self._configured:
      _audit({'method': 'send_file', 'file': {'path': file_path, 'name': file_name},
              'skipped': 'not_configured'})
      return
    file_name = file_name or os.path.basename(file_path)
    try:
      with open(file_path, 'rb') as f:
        file_content = f.read()
      upload_req = (CreateFileRequest.builder()
                    .request_body(CreateFileRequestBody.builder()
                                  .file_name(file_name).file_type("stream")
                                  .file(io.BytesIO(file_content)).build()).build())
      upload_resp = self.client.im.v1.file.create(upload_req)
      if not upload_resp.success():
        lark.logger.error(f"文件上传失败: [{upload_resp.code}]{upload_resp.msg}")
        _audit({'method': 'send_file', 'file': {'path': file_path, 'name': file_name},
                'ok': False, 'response': {'code': upload_resp.code, 'msg': upload_resp.msg}})
        return upload_resp
      file_key = upload_resp.data.file_key
      msg_req = (CreateMessageRequest.builder()
                 .receive_id_type('chat_id')
                 .request_body(CreateMessageRequestBody.builder()
                               .receive_id(LARK_RECEIVE_ID).msg_type("file")
                               .content(lark.JSON.marshal({"file_key": file_key})).build())
                 .build())
      res = self.client.im.v1.message.create(msg_req)
      return self._log_and_audit(
          res, {'method': 'send_file',
                'file': {'path': file_path, 'name': file_name, 'file_key': file_key},
                'size_bytes': len(file_content)})
    except Exception as e:
      _audit({'method': 'send_file', 'file': {'path': file_path, 'name': file_name},
              'ok': False, 'error': repr(e)})
      lark.logger.error(f"send_file 异常: {e}")

  # ── 简易通知卡片（v2 schema 自然升级） ─────────
  def send_notification_card(
      self,
      content: str,
      level: LarkMsgLevel = LarkMsgLevel.Success,
      title: str = None,
      sub_title: str = None,
  ):
    elements = []
    if sub_title:
      elements.append(md_div(f"**{sub_title}**"))
    if content:
      elements.append(md_div(content))
    card = make_v2_card(title=title or '', level=level, elements=elements)
    return self.send_card(card)

  # ── 表格卡片（v2 schema 原生 table） ────────────
  def send_table_card(
      self,
      *,
      title: str,
      level: LarkMsgLevel = LarkMsgLevel.Info,
      subtitle: str | None = None,
      summary_md: str | None = None,
      tables: list[dict] | None = None,
      footer_md: str | None = None,
  ):
    """通用「标题 + 概要 + 多张表格 + 脚注」卡片。

    Args:
        title: 主标题
        level: 颜色等级
        subtitle: 副标题（紧贴在标题下）
        summary_md: 顶部 lark_md 概要文本（可选）
        tables: 表格列表，每项 {'title': 标题, 'columns': [...], 'rows': [...], 'element_id': 'xxx'}
        footer_md: 底部 lark_md 脚注（可选）
    """
    elements: list[dict] = []
    if summary_md:
      elements.append(md_div(summary_md))
    for i, t in enumerate(tables or []):
      if t.get('title'):
        if elements:
          elements.append({'tag': 'hr'})
        elements.append(md_div(t['title']))
      elements.append(make_v2_table(
          columns=t['columns'], rows=t['rows'],
          element_id=t.get('element_id', f'tbl_{i}'),
          page_size=t.get('page_size', 10),
          freeze_first_column=t.get('freeze_first_column', True),
      ))
    if footer_md:
      elements.append({'tag': 'hr'})
      elements.append(md_div(footer_md))
    card = make_v2_card(title=title, level=level, subtitle=subtitle, elements=elements)
    return self.send_card(card)


lark_sender = LarkSender(LARK_APP_ID, LARK_APP_SECRET)


if __name__ == "__main__":
  lark_sender.send_notification_card(
    level=LarkMsgLevel.Danger,
    title="这是标题",
    content="这是内容",
  )
