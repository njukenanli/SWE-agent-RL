# RL with SWE-agent
This repo provides minimal framework to sample SWE-agent trajectories and RL on the trajs.

View [docs/demo.pdf](./docs/demo.pdf) for detailed algorithm explanation.

## On CPU side
```bash
cd cpu
cd sweagent && python -m pip install --upgrade pip && pip install --editable .
```

```bash
python main.py --parallel_instances 1 --dataset dataset.jsonl
```

The code waits until GPU side is ready (receives ACK from GPU side) and start rollout.

## On GPU side


```bash
cd gpu
```

We uses a vllm base image for env setup in gpu/verl/README.md

```bash
bash main.sh --epoch 120 --data-dir /Data
```

The code starts vllm, until CPU side sends ACK -- rollout ends and rsync trajs to GPU. Then it starts verl training. When training ends the vllm is started again from the updated LM weights.
