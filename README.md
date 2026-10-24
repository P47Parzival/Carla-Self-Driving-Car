# Carla-Self-Driving-Car
This project trains a self-driving car in CARLA using SAC reinforcement learning. It features a custom CNN with residual blocks and attention, reward shaping (speed, collision, smoothness), Arduino real-time feedback, hyper parameter tuning, check-pointing, and logging.


This project is about training a simulated self-driving car using reinforcement learning. Here’s a simple breakdown:

  1)Simulated Driving: It uses the CARLA simulator to create a realistic driving environment complete with traffic, weather changes, and various obstacles.

  2)Learning to Drive: The car learns to drive using a method called Soft Actor-Critic (SAC), which is a popular reinforcement learning algorithm.

  3)Custom Features: The system processes camera images and other state information using a custom neural network. This network uses convolutional layers with residual blocks and an attention mechanism to better understand what’s happening     on the road.

  4)Reward System: The car receives rewards or penalties based on factors like distance traveled, maintaining a target speed, avoiding collisions, and smooth driving.

  5)Real-time Feedback: The project also integrates with an Arduino board to send real-time data (like speed and rewards) to external displays.

  6)Training Enhancements: It includes features like hyperparameter tuning (using optuna), checkpointing (to save progress), and detailed logging to track how the training is going.

------------------------------------------------------------------------------------------------------------------------------------------------------

Videos showing the project:

https://www.youtube.com/watch?v=uS3r_8r4riY&t=10s

https://www.youtube.com/watch?v=lIOpiagK0PU

------------------------------------------------------------------------------------------------------------------------------------------------------

For beginners to use the project - 

1) download all the files and remember the folder name where you have the checkpoint 

2) Download Carla-0.10.0 and run it (REMEMBER TO DISABLE RAY TRACING FOR BETTER PERFORMANCE! THROUGH THE CONFIG)

3) In the folder having train_sac_carla.py and carla_env.py. type cmd in the folder's address and paste 
python train_sac_carla.py --resume "YOUR_CHECKPOINT_ADDRESS"\sac_carla_160000_steps.zip

if you want to train it from starting just remove --resume 

#### Run hyperparameter optimization before training:

python train_sac_carla.py --optimize


Make sure you're in the correct directory where the script is located when running these commands.

------------------------------------------------------------------------------------------------------------------------------------------------------

Features for futures release: (for contributors <3)

1) making the reward system better and better training features

2) More training matrices for visualization

------------------------------------------------------------------------------------------------------------------------------------------------------

## ☁️ Running this on AWS (cloud GPU training)

Since training needs a real GPU and CARLA itself is heavy to run locally, this project can be trained on an AWS EC2 GPU instance instead. Everything needed for this lives in the `aws/` folder. This is software-only (no Arduino)  the Arduino integration in `carla_env.py`/`train_sac_carla.py` is already optional and auto-disables when `pyserial` isn't installed.

### Architecture

- **One EC2 GPU instance** (`g5.2xlarge`, NVIDIA A10G, 24GB VRAM) runs both the CARLA server and the training script together  they talk over `localhost:2000`, so there's no cross-machine network latency.
- **S3 bucket** backs up checkpoints and TensorBoard logs every 15 minutes via a cron job, so progress survives a stopped/terminated instance.
- Everything runs inside a **tmux session** (`carla-rl`) so it keeps running after you disconnect your SSH session. Windows:
  - `0: carla`  the CARLA server itself
  - `1: train`  the training script, live logs
  - `2: tensorboard`  TensorBoard, served on port 6006
  - `3: live`  a live MJPEG chase-camera view of the car, served on port 8082
  - `4: lidar`  a live radar-style LIDAR bird's-eye view (same 180° front-facing sensor the agent trains on), served on port 8083

### One-time setup (already done, documented for reference)

1. AWS CLI profile `carla-rl` configured with an IAM user that has EC2/S3 permissions (kept separate from any other AWS credentials on this machine).
2. `aws/provision` steps created: an SSH key pair (`aws/carla-rl-key.pem`), a security group restricted to your IP on port 22 only, and an S3 bucket for checkpoint backups.
3. `aws/remote_setup.sh`  run once per fresh instance. Downloads CARLA 0.10.0 (Linux server build), creates a **Python 3.12 venv** (CARLA 0.10's bundled Python API wheel requires 3.12, not the system's default 3.10), and installs all training dependencies into it  nothing installed globally.
4. **GPU driver gotcha (important if you ever recreate the instance):** the AWS Deep Learning AMI ships NVIDIA's compute-only ("Tesla") driver, which has no real GPU-rendering path  CARLA's renderer crashes on it. You must install the **NVIDIA GRID driver** instead (`s3://ec2-linux-nvidia-drivers/latest/`, see AWS's "Install NVIDIA GRID drivers" doc) and reboot the instance afterward. Confirmed working combo: GRID driver 595.91.07 on a `g5.2xlarge` (A10G).
5. **cuDNN gotcha:** the same AMI also has a system-wide CUDA/cuDNN install that can shadow the venv's own PyTorch-bundled cuDNN via `LD_LIBRARY_PATH`, causing an intermittent `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` crash after a few hundred training steps. Fixed by launching training with `LD_LIBRARY_PATH` unset  already baked into `run_training.sh` and `restart_train.sh` (`env -u LD_LIBRARY_PATH python train_sac_carla.py`).

### Day-to-day commands (run from this folder, on your local machine)

| Task | Command |
|------|---------|
| Start the (stopped) instance | `bash aws/start.sh`  gets a new public IP automatically |
| Deploy code changes + install CARLA/deps (first time or after code changes) | `bash aws/deploy.sh` |
| Launch CARLA + training + TensorBoard + live views (auto-resumes from the latest checkpoint if one exists) | `ssh -i aws/carla-rl-key.pem ubuntu@<ip> "~/carla-rl/run_training.sh"` |
| Restart just the training process after a crash (also auto-resumes) | `ssh -i aws/carla-rl-key.pem ubuntu@<ip> "~/carla-rl/restart_train.sh"` |
| Watch live (TensorBoard + chase-cam + LIDAR radar, all in your browser) | `bash aws/connect.sh`, then open `localhost:6006`, `localhost:8082`, `localhost:8083` |
| Grab a one-off short video clip of the car driving | `python aws/spectate.py 15` (run on the instance), then `ffmpeg` the resulting frames into an mp4 |
| Stop the instance (saves cost, keeps all data) | Gracefully stop training first (`Ctrl+C` in the `train` tmux window), then `bash aws/stop.sh` |
| Fully tear down everything (keeps your S3 checkpoints) | `bash aws/teardown.sh` |
| Re-authorize SSH if your home IP changes | `bash aws/update_ip.sh` |

### Known AWS quirk: GPU capacity shortages

`g5.2xlarge` on-demand capacity in `ap-south-1` can run out entirely (both AZs) at times  `aws ec2 start-instances` fails with `InsufficientInstanceCapacity`. This isn't caused by stopping the instance "losing" anything  stopped On-Demand instances simply release their underlying host and need to find new capacity on restart, which is a known limitation for GPU instance types. If you hit this, just retry periodically (it clears unpredictably, from minutes to hours)  do **not** downgrade to a smaller GPU (e.g. `g4dn.xlarge`/T4) as a fix, since it has half the VRAM and was causing OOM-adjacent issues during training.

### Env vars needed

Only `aws/.env` (copy from `aws/.env.example`)  AWS credentials for the S3 checkpoint sync. No other API keys are required; the project doesn't call any third-party/model APIs.


