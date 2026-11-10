# 第 13 步：部署到 Ollama

最后一步。装进 Ollama，就能像普通模型一样用。

## 前置

Ollama 要在运行（`ollama --version` 能出版本就行，没启动就先开一下）。

## 一条命令

在 `learn/` 目录下：

```powershell
..\.venv\Scripts\python.exe pipeline\deploy_ollama.py
```

它做三件事：按 `config/persona.yaml` 写 `Modelfile` → `ollama create` 建模型 →
发 5 句真实聊天测试，能答上来才算完。

跑完看 `work/reports/13-deploy.md`。

## 模型名和参数

都在 `config/persona.yaml` 的 `deploy:` 段里改：

```yaml
deploy:
  model_name: persona-a   # 只能小写字母、数字、- _ .
  num_ctx: 4096
  temperature: 0.7
  top_p: 0.9
  repeat_penalty: 1.1
  num_predict: 160                # 单次回复上限，防长篇
```

`num_predict` 别调大。这个框架的目标是一次发一两句，放开了就会变回助手腔。

## 怎么用

```powershell
ollama run persona-a
```

接进自己的程序：

```powershell
curl http://127.0.0.1:11434/api/chat -d '{"model":"persona-a","messages":[{"role":"user","content":"在吗"}]}'
```

## 系统提示已经写进模型了

`Modelfile` 里的 `SYSTEM` 段就是训练时用的那一份（只接话、不清楚就问、一次一两句），
不用每次调用再传，也别在调用时覆盖它——覆盖了内容一致性就没人保证了。

## 出问题看这里

| 现象 | 原因 |
|---|---|
| `invalid model name` | `Modelfile` 的 `FROM` 写成了相对路径，脚本已改成绝对路径 |
| 创建成功但回复很长 | 调用时覆盖了 `SYSTEM`，或者把 `num_predict` 调大了 |
| 回复全是助手腔 | `SYSTEM` 没生效，检查 `ollama show persona-a --modelfile` |
| 找不到模型 | `ollama list` 确认名字，模型名不能有大写字母 |
