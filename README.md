# RL Robot Navigation

## 安装依赖

```bash
conda create -n rl_dune python=3.10 -y
conda activate rl_dune
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy cvxpy pyyaml ir-sim gymnasium

```

## DUNE 模型训练

```bash
python dune/dune_train.py --config config/dune_train.yaml
```

## TD3 强化学习训练

```bash
python train_td3.py
```

## TD3 策略测试

```bash
python test_td3.py
```

## 环境测试

```bash
python train_env_test.py
```
