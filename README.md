# AIPI 591: Lab 5: MiniGrid Dynamic Obstacle 8x8

### Author: Matana Pornluanprasert

Train the Agent on 8x8 MiniGrid with the following reward structure.

---

## Reward Function

The agent is trained with PPO on `MiniGrid-Dynamic-Obstacles-8x8-v0`. The default MiniGrid reward is sparse: **0** on most steps, a success bonus of `1 - 0.9 * (step_count / max_steps)` for reaching the green goal, **-1** (and episode end) for colliding with an obstacle or a wall, and **0** on timeout. That made waiting until timeout safer than exploring.

`RewardWrapper` in `main.py` changes two things:

1. **Walls are not fatal.** Walking into a wall no longer ends the episode with **-1**. Hitting a moving ball is still **-1** and terminates.
2. **Step penalty of `-0.01`.** Every action costs a little, so standing still until timeout is no longer free.

| Event | Reward |
| ----- | ------ |
| Reach the goal | `1 - 0.9 * (step_count / max_steps)`, minus `0.01` for that step |
| Hit a moving obstacle (ball) | `-1.01`, episode ends |
| Walk into a wall | `-0.01`, episode continues |
| Other steps | `-0.01` |
| Timeout | accumulated step penalties only (no success bonus) |

---

## Files

```
main.py              PPO training, evaluation, video recording, RewardWrapper
test_ppo.ipynb       Notebook that runs train(), evaluate(), record_video()
requirements.txt     Python dependencies
model.zip            Saved PPO policy
demo.mp4             Recorded evaluation rollout
logs/                TensorBoard training logs
```

---

## How to Run

**Install dependencies** (use the Lab 5 virtual environment):

```
pip install -r requirements.txt
```

**Train, evaluate, and record a video:**

```
python main.py
```

Or open `test_ppo.ipynb`, and run the cells in order.

**View training curves:**

```
tensorboard --logdir logs
```
