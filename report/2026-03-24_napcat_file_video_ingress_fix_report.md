# Napcat 视频文件入站修复报告（2026-03-24）

## 问题
- 用户发送 `.mp4` 时，Napcat 入站段是 `file` 而不是 `video`。
- 结果消息链路只显示 `[文件:xxx.mp4]`，不会触发视频摘要。

## 根因
- `napcat_adapter` 的 `file` 分支此前固定返回 `type=file`。
- 核心视频摘要链路仅在 `type=video` 时触发。

## 实施修复
文件：`napcat_adapter/src/handlers/to_core/message_handler.py`

1. 新增视频后缀识别（`.mp4/.mov/.webm/...`）。
2. `file` 分支支持传入 `raw_message`，用于补拉详情。
3. 对视频文件执行分层补全：
   - 优先从 file 段提取 `url/path/base64`；
   - 不足时补拉 `get_msg`；
   - 仍不足时通过 `file_id` 调用 `get_file / get_private_file_url`（群聊再试 `get_group_file_url`）。
4. 只要拿到 `base64/url/path` 任一来源，就转成 `video` 段进入视频链路。
5. `_handle_video_message` 新增 `base64` 直通处理，避免二次下载。

## 关键日志
- `检测到视频文件扩展名，按视频链路处理`
- `检测到视频文件扩展名，已通过补全拿到 base64，按视频链路处理`
- `视频文件 File API 补全失败：未获取到 base64/url/path`

## 验证
- 已通过 `py_compile`：
  - `napcat_adapter/src/handlers/to_core/message_handler.py`

