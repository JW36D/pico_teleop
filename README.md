# PICO 双臂遥操与数据采集

## 启动采集

在 VS Code 或 SSH 终端运行：

```bash
cd pico_teleop
bash run_collection.sh --execute
```

自动左右分屏：左侧遥操，右侧显示录制状态和操作记录。

启动前完成头显设置：打开 Send，关闭 Switch w/ A Button。每次新启动后，先戴正头显、朝前，按既有方法完成首次 Grip 接管。之后松开 Grip，可将头显挂脖子，直接看实机操作，保持手柄追踪有效。

| 操作 | 功能 |
| --- | --- |
| Grip | 接管对应机械臂；松开停止该侧遥操 |
| 按住 Grip 后操作扳机 | 控制对应夹爪开合 |
| A | 开始／结束一条录制，独立于 Grip |
| X | 结束录制后删除最新一条；重复按不会继续删更早记录 |
| B | 切换头显显示 |
| B+Y 同时按，或左侧终端 Ctrl+C | 退出遥操采集并收尾 |

可以在完全静止、持续按 Grip、刚松 Grip 尚在减速时按 A 结束录制。首次接管及松开后的正常过渡已有对应转换处理，不需要为录制专门松开、重按 Grip，也不要求左右夹爪先各开合一次。

需要看腕部画面时，在头显选择 ZEDMINI 并开启 Listen，地址沿用当前 Jetson 地址。Listen 可以按需开关，录制不依赖头显是否显示画面。**视频独立于采集程序启动仍是后续待办。**

## 查看结果与转换数据集

退出后，父终端显示本次录制、删除、保留、完整保存和可转换的条数，以及 raw 路径和本次两种格式的转换命令。复制对应命令即可导出，无需先手工转视频。

原始数据位置由 `configs/collection.json` 中的 `data_root` 设置。当前相对路径为：

```text
../../data/pico_raw/sessions/<本次会话>/
```

其中 `collection_summary.txt` 可随时查看，`episodes/` 保存保留的记录，`.trash/` 保存已删除记录。原始画面按 JPG 帧及时间戳保存；LeRobot 导出时自动生成视频、状态和动作数据表。

重新查看指定会话的汇总：

```bash
bash ./run_collection.sh --summary /本次raw会话目录
```

“完整保存”表示原始记录完整；“可转换”表示通过本项目转换检查，尚不能代替图像视野和实际任务质量验收。

## 其他入口

```bash
# 仅演示终端界面，不连接机械臂、相机或头显
bash ./run_collection.sh --demo
```

普通遥操、接收服务启动、回位及检查工具集中在仓库内的 [pico_teleop_dev](pico_teleop_dev/README.md)。已有接收服务可以继续使用。

## 目录

| 目录 | 内容 |
| --- | --- |
| `run_collection.sh` | 启动遥操采集、查看会话汇总 |
| `export_dataset.sh` | 导出 LeRobot 数据集 |
| `configs/` | 相机模板、采集、左右臂和导出配置；复制 `cameras.example.json` 为 `cameras.json` 并填写本机相机序列号 |
| `app/` | 运行代码；诊断日志在 `app/state/` |

日常运行不依赖开发目录中的测试和工具。当前控制Python仍使用相邻 `RM/.venv/bin/python`，接收器沿用已有服务。部署路径、转换数据语义和验证范围见[开发与迁移说明](pico_teleop_dev/docs/开发与迁移说明.md)。
