# PhyRL: Physics-Informed Model-Based Reinforcement Learning

Usage:

- **Train baseline, Dreamer v2**
  - Get latest Dreamer v2 code: https://github.com/danijar/dreamerv2
  - Run ```python lunarlander_train.py``` on Dreamer v2 repository
- **PhyRL: Train Model**
  - Go to ```examples/lunarlander```
  - Run ```python phyrl.py [Physics Constraint Index]``` where the index ranges from 1 to 4 as presented in the paper report.
  - Observe the results on Weights & Biases: https://wandb.ai/lucascamara/PhyRL
- **PhyRL: Train RL Agent**
  - Go to ```examples/lunarlander```
  - Run ```python phyrl_agent.py [Physics Constraint Index]``` where the index ranges from 1 to 4 as presented in the paper report.
    - Note: The RL agent will randomly pick one learned model from Weights & Biases corresponding to the desired index, and then train on it.
  - Observe the results on Weights & Biases: https://wandb.ai/lucascamara/PhyRL_Agents

Packages:

```
python
numpy
scipy
pysindy
scikit-learn
torch
tqdm
pandas
```
