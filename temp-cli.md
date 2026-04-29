## 测试推理

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"你好，请用一句话介绍你自己"}],"max_tokens":100}'
```
