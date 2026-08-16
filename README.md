# RL with SWE-agent
This repo provides a minimal framework to sample SWE-agent trajectories and do RL on the trajs.

View [docs/demo.pdf](./docs/demo.pdf) for detailed algorithm explanation.

## Environment setup

### On CPU side

The CPU machine must be a physical machine or AWS VM that allows running docker inside. Containerized pods cannot run docker.

Minimun cpu machine requirements: 64 cpus, 500GB RAM, 8TB storage.

```bash
cd cpu
python -m venv venv
source venv/bin/activate
cd sweagent
python -m pip install --upgrade pip && pip install --editable .
```

### On GPU side

The GPU workflow targets one x86-64 machine with 8 NVIDIA B200 GPUs. 

```bash
docker pull verlai/verl:vllm017.latest

docker run --rm -it \
  --gpus all \
  --net=host \
  --ipc=host \
  --shm-size=512g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -v "$PWD/gpu:/workspace/gpu" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/gpu \
  --name verl-rft-rl \
  verlai/verl:vllm017.latest \
  bash
```

Inside the container, install the repository's VERL checkout and verify the
training and serving dependencies:

```bash
cd /workspace/gpu/verl
pip install --no-deps -e .

python - <<'PY'
import torch
import pandas
import pyarrow
import megatron.core
import transformer_engine
import vllm

print("cuda:", torch.version.cuda)
print("gpus:", torch.cuda.device_count())
print("vllm:", vllm.__version__)
PY

cd /workspace/gpu
```

If the image is missing the Megatron dependency stack, install it before the
editable VERL install:

```bash
cd /workspace/gpu/verl
USE_SGLANG=0 USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd /workspace/gpu
```

## Run

### On CPU side

```bash
python main.py \
  --num_workers 1 \
  --train_set train.jsonl \
  --test_set test.jsonl \
  --batch_size 64
```

The sweagent side waits until GPU side is ready (receives ACK from GPU side) and start rollout.

`--batch_size` is the number of training task instances rolled out in each
epoch. The selection advances through `--train_set` by epoch and wraps at the end.
The selected instances overwrite `cpu/sweagent/temp.jsonl`, which is used only
for training rollouts. Test rollouts use the complete `--test_set` in every epoch.
Keep the training batch smaller than the full dataset so each epoch performs
one RL mini_batch=full_batch=one weight-update step -- this is necessary for agentic RL.

## On GPU side

In side the container specified above:

```bash
bash main.sh --epoch 120 --data-dir /Data
```

The code starts vllm, until CPU side sends ACK -- rollout ends and rsync trajs to GPU. Then it starts verl training. When training ends the vllm is started again from the updated LM weights.
