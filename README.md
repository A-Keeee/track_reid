
# Track Torch 项目启动说明

## 直接启动脚本

在项目根目录下运行以下命令启动服务：

```bash
bash start.sh
```

## 设置为系统自启动服务

1. 复制 `self_start` 文件夹到主目录（以 `/home/juwei/` 为例）：

    ```bash
    sudo cp -r self_start /home/juwei/
    ```

2. 将 `self_start` 文件夹中的 `track.service` 复制到 systemd 服务目录：

    ```bash
    sudo cp /home/juwei/self_start/track.service /etc/systemd/system/
    ```

3. 重新加载 systemd 服务配置：

    ```bash
    sudo systemctl daemon-reload
    ```

4. 启动服务：

    ```bash
    sudo systemctl restart track.service
    ```

5. 设置服务开机自启动：

    ```bash
    sudo systemctl enable track.service
    ```

---

如有疑问请联系项目维护者。


