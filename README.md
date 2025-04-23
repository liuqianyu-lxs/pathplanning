 文件结构：
├── main.py                           # 主入口：支持四种模型组合训练

├── run_all_models.py                # 自动批量训练 A1/A2/B1/B2 模型

│

├── agent/

│   ├── dqn_agent.py                 # 普通 DQN 实现

│   └── ddqn_agent.py                # DDQN 实现

│

├── models/

│   ├── qnet_cnn.py                  # CNN 

│   └── qnet_mlp.py                  # MLP 

│

├── env/

│   └── map_env.py                   # 环境类 MapEnv + 状态提取函数

│

├── utils/


│   ├── logger.py                    # 日志/模型/轨迹/图片保存等

│   ├── map_utils.py                 # 障碍地图和监控盲区生成

│   └── config.py                    # 空文件

│

├── core/

│   └── train_utils.py               # 训练逻辑 + 评估函数 + 图表生成

│

└── results_*                        # 每次训练自动生成的结果输出目录



| model | Algorithm | Network |
|------|-----------|---------|
| A1   | DQN       | MLP     |
| A2   | DQN       | CNN     |
| B1   | DDQN      | MLP     |
| B2   | DDQN      | CNN     |
2023-04-22/23
当前版本：
每600轮训练提高一次地图难度
CNN远好于MLP
存在问题：
地图难度上限设置过高，3000 轮之后训练效果下降太多
没设置 reward 下限，最大步数设置太高，打转的地图中浪费时间。
下一版：
调低地图难度上限，3000 轮之后效果应该可以问题，稳定后理论上 DDQN+CNN 最佳。
