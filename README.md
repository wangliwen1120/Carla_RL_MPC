### 1. 环境配置
可配置虚拟环境或者Virtualenv。如果配置Virtualenv，会在Carla_Gym文件夹中产生文件夹Carla_RL_env。

按照requirements.txt按照依赖。

### 2. setup

开启终端，启动CarlaUE4.sh：


`./CarlaUE4.sh -carla-server -fps=20 -world-port=2000 -windowed -ResX=1280 -ResY=720 -carla-no-hud -quality-level=Low`

如果不想可视化，可使用

`DISPLAY= ./CarlaUE4.sh -opengl`

（但没感觉快多少）

运行test.py

