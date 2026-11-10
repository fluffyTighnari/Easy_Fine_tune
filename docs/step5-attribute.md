# 第 5 步：说话人归因与会话切分

**产出**：`work/data/attributed.jsonl`、`work/reports/5-attribute.md`

## 检查配置

`config/persona.yaml`（第 2 步复制的那份）：

```yaml
persona:
  name: 人物A
  self_names: [人物A]        # 必须和 chat.txt 里的说话人一字不差
  system_prompt: 你正在用微信和朋友聊天。

data:
  session_gap_min: 30         # 间隔超这么多分钟就切成新会话
  max_burst: 8                # 连发超过这么多条就截断
```

## 跑

```powershell
cd learn
..\.venv\Scripts\python.exe pipeline\attribute.py
```

## 做了三件事

| 做什么 | 规则 |
|---|---|
| 归因 | `sender` 在 `self_names` 里 → `me`；其余自然人 → `them`；系统 → `system` |
| 会话切分 | 相邻消息间隔 > `session_gap_min` 就开新会话 `sid` |
| 轮次 | 会话内角色变化一次 `turn + 1`；系统消息不占轮次 |

## 报告看什么

| 项 | 要判断的 |
|---|---|
| `me` 占比 | 10%~90%；太低说明导出的是对方的话 |
| `me` 条数 | ≥ 300，否则第 7 步样本不够 |
| 会话数 / 每会话消息数 | p50 太小说明对话很碎 |
| 连发 p90 / max | 第 7 步按 `max_burst` 截断，max 太大会丢内容 |

## 闸门

| 闸门 | 标准 |
|---|---|
| `me` 占比 | 10%~90% |
| `me` 条数 | ≥ 300 |
| 有会话切分 | ≥ 1 |

## 出问题怎么办

| 现象 | 处理 |
|---|---|
| `me` = 0 | `self_names` 写错了，抄 chat.txt 里的原文 |
| `me` 占比 < 10% | 换个人学，或重新导出含自己发言的记录 |
| 出现 `group` 标记 | 有 3 个以上自然人，先确认要不要按群聊训 |
| 会话数过少（比如 1 个） | `session_gap_min` 调小，比如 10 |
| 会话数过多、p50 只有 1~2 条 | `session_gap_min` 调大，比如 60 |

## 下一步

第 6 步：清洗（`pipeline/clean.py`）

## 名字对不上怎么办

配置里的 `self_names` 在数据里一个都没出现时，这一步会停下并列出数据里的说话人：

```
配置里的 self_names=['人物A'] 在数据里一个都没出现。
数据里的说话人：人物B（8,376 条）、人物A（5,453 条）
```

两种改法，选一个：

```powershell
# 临时指定，不改配置文件
..\.venv\Scripts\python.exe pipeline\attribute.py --as 人物A

# 或者直接改 config/persona.yaml 的 self_names
```

数据做过匿名化、或者配置是从示例复制的，就会遇到这个。
