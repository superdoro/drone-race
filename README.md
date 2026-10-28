# Drone Race：用 RL 學「最快飛到目標 + 自動避障」

用 [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) 模擬器同時跑上千架 Crazyflie（cf2x），再用 [rsl-rl](https://github.com/leggedrobotics/rsl_rl) 的 PPO 訓練。
目前這個階段，AI 直接知道障礙物的位置。架構上已經把「任務邏輯」和「模擬器」分開，之後要換成 Isaac Lab 或深度相機，都不用重寫任務。

## 安裝

```bash
conda create -n drone python=3.12 -y
conda activate drone
pip install torch            # CUDA 版；RTX 50 系列（Blackwell）需要 CUDA 12.8 以上的 build
pip install -r requirements.txt
```

## 使用

```bash
python check_env.py                                   # 環境自我檢查：shape / NaN、懸停、角速度追蹤、柱子位置、撞柱子和撞牆判定
python train.py --num_envs 4096 --max_iterations 3000 --exp_name corridor  # 訓練，結果存到 logs/corridor/<時間>/
tensorboard --logdir logs                             # 看成功率、碰撞率、到達時間、平均速度、curriculum 等級
python eval.py --run logs/corridor/<時間>              # 用最難的等級評估
python eval.py --run logs/corridor/<時間> --level 2 --record   # 另外錄一段 env 0 的 mp4
python eval.py --run logs/corridor/<時間> --obstacles 8 8 --dist 10 12   # 測試特定條件
python eval.py --run logs/corridor/<時間> --viewer --num_envs 1
```

多 GPU 的電腦（例如 2 張 5080）：用 `CUDA_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=1` 指定 GPU，一張卡跑一組實驗；也可以把 `--num_envs` 開大。

## 任務設計

| 項目 | 內容 |
|---|---|
| 任務 | 每個 episode 隨機產生起點、終點（方向任意），並圍出一條 3 m 寬的封閉走廊：兩側有牆，起點後 1 m、終點後 1.5 m 也各有一道牆。走廊裡放 K 根圓柱，沿走廊方向分段排列，橫向可以出現在任何位置，包括貼著牆。牆和柱子都比天花板高，所以只能穿過柱子之間，沒辦法繞外圈或從上方飛過 |
| 動作（4 維，範圍 [-1, 1]） | **CTBR**：總推力（0 = 懸停推力）+ 三軸角速度指令。由 [control/ctbr.py](control/ctbr.py) 的角速度控制器和 mixer 換算成四顆馬達的推力。Crazyflie、PX4、Betaflight 都吃這個介面 |
| 觀測（65 維，都在機體座標系） | 目標方向和距離、線速度、角速度、重力方向、上一步的動作、最近 8 根柱子的相對位置和表面距離、4 面牆的最近點方向和距離 |
| 獎勵 | 往目標前進的距離 × 5、每一步扣 0.02（越快到越好）、角速度和動作變化的懲罰、yaw 旋轉的懲罰（不然 AI 會一直原地打轉）、輸出超出 [-1, 1] 的懲罰（避免變成 bang-bang 控制）、離柱子或牆太近的懲罰、撞到 −40、到達 +10 |
| 結束條件 | 到達（距離 < 0.3 m）、撞到柱子、撞牆、撞地面或飛出天花板、超過 15 秒 |
| Curriculum | 5 個等級：第 0 級是 0 根障礙物、2–4 m；之後每升一級就把範圍往上加，最高到 0–8 根柱子、2–12 m（走廊太短時會自動減少柱子數量，確保一定有路可以過）。範圍只放寬、不平移，所以簡單的情況一直都在訓練裡（不然 AI 會忘記短距離要怎麼煞車）。成功率 ≥ 80% 就升級 |
| 頻率 | 物理和角速度控制 100 Hz，策略 50 Hz |

所有參數都在 [config/task_cfg.py](config/task_cfg.py)。

## 基礎訓練結果（2026-10-29，RTX 3050 Ti 4GB）

### 第一版：開放場地（沒有圍牆）

這一版的 AI 主要是從柱群外側繞過去，並沒有真正學會在柱子之間閃避，所以後來加上了圍牆。

訓練設定：4096 個環境、2000 iterations，總共約 1.97 億步，花了約 17 分鐘，顯示記憶體大約用到 0.8 GB。Run 名稱是 `logs/goal_nav/2026-10-29_00-30-24`。
評估：用 deterministic policy，每種條件約 2000 個 episode。

| 條件 | 成功率 | 撞到障礙物 | 平均到達時間 | 平均速度 |
|---|---|---|---|---|
| 0 根，2–4 m | 100% | 0% | 0.64 s | 4.6 m/s |
| 2 根，3–6 m | 95.9% | 3.5% | 0.98 s | 4.7 m/s |
| 4 根，4–8 m | 98.8% | 1.0% | 1.22 s | 5.0 m/s |
| 8 根，6–12 m | 96.3% | 3.4% | 1.40 s | 6.5 m/s |
| 混合（0–8 根，2–12 m） | 98.0% | 1.9% | 1.20 s | 5.8 m/s |

### 第二版：封閉走廊（目前的版本）

訓練設定：4096 個環境、3000 iterations，約 27 分鐘。Run 名稱是 `logs/corridor/2026-10-29_02-19-56`（感測 8 根柱子、感測距離 8 m、撞到 −40）。

| 條件 | 成功率 | 撞柱子 | 撞牆 | 平均到達時間 | 平均速度 |
|---|---|---|---|---|---|
| 0 根，2–4 m | 99.5% | 0% | 0.5% | 0.66 s | 4.5 m/s |
| 2 根，3–6 m | 95.6% | 4.4% | 0% | 0.88 s | 5.1 m/s |
| 4 根，4–8 m | 77.3% | 22.3% | 0.4% | 1.11 s | 5.4 m/s |
| 8 根，6–12 m | 39.8% | 52.8% | 6.3% | 1.50 s | 5.8 m/s |
| 混合（0–8 根，2–12 m） | 79.7% | 18.6% | 1.6% | 1.05 s | 6.0 m/s |

加了圍牆以後，AI 確實得在柱子之間閃避，但柱子一密成功率就明顯下降。分析撞擊的瞬間：撞擊時的前進速度中位數大約 10 m/s，而且大多是正面撞上。也就是說，AI 還沒學會遇到密集區要先減速。
把感測從 4 根加到 8 根、感測距離從 4 m 拉到 8 m、撞擊懲罰從 −20 加到 −40，都只讓成功率提升 1–2%。

注意：Genesis 的無人機模型沒有空氣阻力，馬達反應也是瞬間的，所以速度會比真的 Crazyflie 樂觀。如果要上實機，要先加這兩項（見下方「之後的方向」）。

## 架構

```
train.py / eval.py ── rsl-rl PPO
        │
task/goal_nav_task.py (VecEnv) ── sampling / corridor / observations / rewards / collision、control/ctbr.py   ← 純 torch，跟模擬器無關
        │  SimBackend 介面（backends/base.py）
backends/genesis_backend.py   ← 唯一會 import genesis 的檔案
```

- 無人機的物理參數集中在 [config/drone_params.py](config/drone_params.py)，所有後端都共用這一份。
- 碰撞（柱子和牆）用 torch 解析計算，不靠模擬器的接觸偵測，所以換模擬器後判定結果完全一樣。

### 移植到 Isaac Lab

1. 新增 `backends/isaaclab_backend.py`，實作 `SimBackend` 的方法（座標慣例寫在 [backends/base.py](backends/base.py) 開頭的說明裡）。`apply_rotor_thrusts` 可以把每顆馬達的推力換算成力和力矩，再用 `set_external_force_and_torque` 套用。
2. 在 `make_backend()` 註冊 `"isaaclab"`。
3. 執行 `python check_env.py --backend isaaclab`，確認懸停、角速度方向和碰撞判定都和 Genesis 一致。
4. 執行 `python train.py --backend isaaclab`。任務、獎勵、控制器和 PPO 設定都不用改。

## 之後的方向

- **換成深度相機**：實作 `SimBackend.get_depth()`（Genesis 用 BatchRenderer，Isaac Lab 用 TiledCamera），把 `obs.mode` 設成 `"depth"`，然後在 `GoalNavTask._compute_obs` 用影像取代障礙物那一段觀測。可以用 rsl-rl 的 distillation，讓這一階段的策略當老師。
- **上實機**：實作 `domain_rand`（質量、推力、延遲），加入空氣阻力和馬達動態，再用 `runner.export_policy_to_onnx` 匯出策略。
