
## On CPU side
```bash
cd cpu
python main.py
```
The code waits util GPU side is ready (receives ACK from GPU side) and start rollout.

## On GPU side
```bash
cd gpu
bash main.sh
```
The code starts vllm util CPU side sends ACK -- rollout ends and rsync trajs to GPU, and then starts verl training. When training ends the vllm is started again from the updated LM weights.
