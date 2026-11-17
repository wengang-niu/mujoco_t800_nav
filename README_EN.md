# mujoco_t800_nav

**Locomotion RL** (reinforcement-learning-based locomotion control) + **VLN** (Vision-and-Language Navigation) for the EngineAI **T800 humanoid robot**, built on [MuJoCo](https://mujoco.org/) simulation.

> The overall project goal is "locomotion RL + VLN navigation": first use RL to teach T800 motor skills such as standing, squat-to-stand, and getting up from the ground, then layer vision-language navigation (VLN) on top. The repository currently contains the locomotion RL part; VLN is still in planning.

## Robot

- Model: EngineAI **T800** (Shenzhen Zhongqing Robot Technology / EngineAI)
- Degrees of freedom: **25** = legs 12 (hip/knee/ankle pitch·roll·yaw ×2) + torso yaw 1 + arms 10 (shoulder/elbow pitch·roll·yaw ×2) + head 2 (pitch·yaw)
- Model source: `models/t800/` (official EngineAI MuJoCo model, **BSD 3-Clause** license)

## Directory Layout

```
mujoco_t800_nav/
├── t800_common.py           # Shared utilities: model loading, joint addressing, PD control, obs/reward physics
├── 01_load_t800_viewer.py   # Load the model and view it
├── 02_control_interface.py  # State/sensor/torque interfaces
├── 03_pd_stand.py           # Hand-written PD standing (non-RL baseline)
├── 04_stand_env.py          # Standing Gymnasium environment
├── 05_ppo.py                # Hand-written PPO (PyTorch, no stable-baselines3)
├── 06_eval_stand.py         # Evaluate the standing policy
├── 07_get_up.py             # "Squat → stand" environment
├── 08_train_getup.py        # Train the squat-to-stand policy
├── 09_eval_getup.py         # Evaluate the squat-to-stand policy
├── 10_get_up_ground.py      # "Ground folded-tuck → squat handover" environment
├── 11_train_getup_ground.py # Train the ground get-up policy
├── 12_eval_getup_ground.py  # Evaluate the ground get-up policy
└── models/t800/             # T800 MuJoCo model (URDF/XML/meshes/textures)
```

## Setup

- Python 3.12, conda environment `t800`
- MuJoCo **4.0**
- Dependencies: `gymnasium`, `numpy`, `torch`

```bash
conda create -n t800 python=3.12 -y
conda activate t800
pip install mujoco gymnasium numpy torch
```

## Quick Start

```bash
conda activate t800

# 1. Load the model and view it
python 01_load_t800_viewer.py

# 2. Hand-written PD standing (non-RL baseline; falls forward in ~1-2s, as expected)
python 03_pd_stand.py --headless

# 3. Standing environment self-check (includes throughput benchmark)
python 04_stand_env.py

# 4. Train the standing policy (default CPU, 2000 iterations)
python 05_ppo.py --train 2000

# 5. Evaluate the standing policy
python 06_eval_stand.py --viewer
```

## Training Pipeline

The project is organized as three progressively harder task stages, each a complete **environment → train → evaluate** loop:

### Stage 1: Standing balance (01–06)

From loading the model and wiring up the control interface, to a hand-written PPO learning to stand on two legs (including disturbance rejection).

| File | Purpose | Run |
| --- | --- | --- |
| `01_load_t800_viewer.py` | Load model + view | `python 01_load_t800_viewer.py` |
| `02_control_interface.py` | Verify state/sensor/torque paths | `python 02_control_interface.py` |
| `03_pd_stand.py` | Hand-written PD standing (non-RL baseline) | `python 03_pd_stand.py [--headless]` |
| `04_stand_env.py` | Standing Gym environment (4-dim action / 39-dim obs) | `python 04_stand_env.py` |
| `05_ppo.py` | Hand-written PPO training | `python 05_ppo.py --train 2000` |
| `06_eval_stand.py` | Evaluate standing / disturbance | `python 06_eval_stand.py [--viewer] [--disturb]` |

### Stage 2: Squat → stand (07–09)

Stand up from a random squat and hold, learned via a four-level curriculum (shallow → deep squat).

| File | Purpose | Run |
| --- | --- | --- |
| `07_get_up.py` | Squat→stand environment (8-dim action / 45-dim obs) | `python 07_get_up.py` |
| `08_train_getup.py` | Training (2000 iterations + curriculum by default) | `python 08_train_getup.py --iters 2000` |
| `09_eval_getup.py` | Evaluate stand-up success rate | `python 09_eval_getup.py --ckpt checkpoints_getup/ppo_getup_final.pt` |

### Stage 3: Ground folded-tuck → squat handover (10–12)

Get up from a ground folded-tuck (the deepest reachable ground pose) to the squat handover band, requiring leg + arm coordination.

| File | Purpose | Run |
| --- | --- | --- |
| `10_get_up_ground.py` | Ground get-up environment (12-dim action / 62-dim obs) | `python 10_get_up_ground.py` |
| `11_train_getup_ground.py` | Training (with curriculum and pre-registered criteria) | `python 11_train_getup_ground.py --iters 2000 [--course auto]` |
| `12_eval_getup_ground.py` | Evaluate handover success rate | `python 12_eval_getup_ground.py --ckpt checkpoints_getup_ground/ppo_gg_final.pt` |

## Key Technical Points

- **Hand-written PPO** (`05_ppo.py`): no stable-baselines3 — a from-scratch shared-trunk Actor-Critic (Gaussian policy + value network), GAE advantage estimation, clipped objective and multi-epoch updates.
- **PD inner loop + RL target angles**: actions output target-angle offsets (standing) or absolute target angles (get-up); a PD controller converts them to torques, running the inner loop at 1kHz physics steps while the policy decides at 100Hz (`frame_skip=5`).
- **Observation/action dimensions**:
  - Standing: 39-dim obs, 4-dim action (ankle pitch/roll only — the verifiable subspace for standing balance)
  - Squat→stand: 45-dim obs, 8-dim action (hip/knee/ankle pitch + ankle roll ×2)
  - Ground get-up: 62-dim obs, 12-dim action (8 leg + 4 arm)
- **Reward & curriculum design**: standing uses upright + height + foot-contact + survival terms; get-up uses height-increment / record-potential progress reward + success-band harvesting + a four-level squat curriculum, with pre-registered criteria (e.g. `value_loss` spikes, cumulative success-band steps) to guard against reward hacking.
- **Vectorization**: the training inner loop uses precomputed addresses (`t800_common.build_addr`) and NumPy fancy indexing to avoid hundreds of string lookups per physics sub-step.

## Notes

- **Model path must be absolute**: relative-path loading fails nondeterministically under MuJoCo 4.0; `t800_common.py` standardizes on absolute paths.
- **Passive viewer exit segfault**: MuJoCo 4.0's passive viewer double-`terminate`s GLFW on exit; scripts end with `time.sleep(0.5); os._exit(0)` to skip the redundant `glfw.terminate` in `atexit`.
- **CPU vs GPU**: this is a single-environment, small-network setup; CPU is empirically faster than GPU. GPU only pays off once you move to multi-environment parallelism with large batches (switch via `--gpu`).
- Checkpoint directories: standing `checkpoints/`, squat-to-stand `checkpoints_getup/`, ground get-up `checkpoints_getup_ground/`.

## License

- Robot model (`models/t800/`): EngineAI, BSD 3-Clause.
- Remaining code: to be determined.
