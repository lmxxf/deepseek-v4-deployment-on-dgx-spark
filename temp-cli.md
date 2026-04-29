```bash
docker login -u lmxxf
```

输入密码（Docker Hub 的密码或 Access Token）。

如果不记得密码，去 https://hub.docker.com/settings/security 生成一个 Access Token 当密码用。

登录成功后：

```bash
docker tag vllm-node-sm120 lmxxf/vllm-deepseek-v4-dgx-spark:latest
docker push lmxxf/vllm-deepseek-v4-dgx-spark:latest
```
