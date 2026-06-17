# RL with SWE-agent
This repo provides minimal framework to sample SWE-agent trajectories and RL on the trajs.

## On CPU side
```bash
cd cpu
python main.py
```

The code waits until GPU side is ready (receives ACK from GPU side) and start rollout.

## On GPU side
```bash
cd gpu
bash main.sh
```

The code starts vllm, until CPU side sends ACK -- rollout ends and rsync trajs to GPU. Then it starts verl training. When training ends the vllm is started again from the updated LM weights.
