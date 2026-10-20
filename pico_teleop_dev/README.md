# PICO 开发与维护材料

日常采集与导出使用仓库根目录的 [pico_teleop](../README.md)。这里保留测试、维护工具、历史配置和依赖构建材料。

```bash
git clone https://github.com/JW36D/pico_teleop.git
cd pico_teleop
cp configs/cameras.example.json configs/cameras.json
cd pico_teleop_dev
```

安装机械臂SDK、相机依赖及接收服务后，将 `PICO_CONTROL_PYTHON` 指向控制环境中的 Python，并按目标机器修改相机序列号、网络和路径配置。

```bash
# 离线回归，不连接设备
bash ./run_tests.sh

# 普通双臂遥操，不做任务录制
bash ./run_teleop.sh --execute

# 检查原始记录
bash ./check_raw_dataset.sh --source /本次raw会话目录

# 只在接收服务未启动时使用
bash ./start_receiver.sh

# 查看已记录的回位姿态，不连接设备
bash ./recover_arm.sh --arm left --offline
```

回位的 `--plan` 读取实机并保存计划，`--execute` 执行回位。回位前检查现场及目标姿态；迁移机器后重新核对，不直接套用旧机器的回位文件。

| 目录 | 内容 |
| --- | --- |
| `tests/` | 回归测试与少量历史基准 |
| `tools/maintenance/` | 普通遥操、回位、夹爪与状态检查 |
| `tools/validation/` | 方向、运动数学、视频及导出验证 |
| `configs/` | 维护和历史测试配置；5份当前配置链接到运行目录 |
| `environment/` | 接收服务源码、镜像构建文件及冻结的 LeRobot 源码归档 |
| `reports/` | 本机生成的验证结果，不纳入公开仓库 |
| `archive/` | 本机保留的精简前备份，不纳入公开仓库 |
| `legacy_recording.py` | 仅供历史回归使用的旧录制器 |

直接运行工具前，在同一个终端载入开发环境：

```bash
source ./paths.sh
"$PICO_CONTROL_PYTHON" "$DEV_DIR/tools/maintenance/status.py" --help
```

开发环境同时导入仓库根目录的 `app/` 和本目录工具；正式入口只导入 `app`，不加载旧录制器或历史调试入口。

现有接收服务和导出镜像继续使用，无需重建。确需重建时，只打包所需构建材料作为上下文：

```bash
cd ..
tar -cf - app/button_events.py pico_teleop_dev/environment/receiver \
  | sudo docker build -f pico_teleop_dev/environment/receiver/Dockerfile -t rm-pico-xr:collection-v1 -

tar -cf - pico_teleop_dev/environment/export \
  | sudo docker build -f pico_teleop_dev/environment/export/Dockerfile \
  --build-arg DATASET_FORMAT=v2.1 \
  --build-arg LEROBOT_COMMIT=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
  -t pico-export:v21 -

tar -cf - pico_teleop_dev/environment/export \
  | sudo docker build -f pico_teleop_dev/environment/export/Dockerfile \
  --build-arg DATASET_FORMAT=v3.0 \
  --build-arg LEROBOT_COMMIT=0b067df57d21d3a02d6c511f1609172fa39ac29b \
  -t pico-export:v30 -
```

更多说明见[开发与迁移说明](docs/开发与迁移说明.md)。
