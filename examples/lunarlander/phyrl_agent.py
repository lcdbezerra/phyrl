import sys
sys.path.append("../../src_torch")
import os
import random
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EveryNTimesteps
from stable_baselines3.common.evaluation import evaluate_policy

from autoencoder import *
from envs_z import *

IBEX = True if os.path.exists("/ibex/scratch/camaral/") else False
online = None
scratch_dir = "/ibex/scratch/camaral/runs/" if IBEX \
    else "/home/camaral/scratch/runs/"
global run_dir, log_dir
run_dir = None
log_dir = None
# base_dir = "/home/camaral/code/SindyAutoencoders/examples/lunarlander/"
dataset_base = "/ibex/scratch/camaral/phyrl/dataset/" if IBEX \
    else "/home/camaral/scratch/phyrl/dataset/"
# ds_name = "dataset_debug" #####
ds_name = "dataset_trained_gstate" #####
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Workaround for using run.config
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_run(phy_constraint):
    api = wandb.Api()
    wpath = "lucascamara/PhyRL"
    runs = api.runs(wpath)
    runs = [run for run in runs if (run.state == "finished") and (run.sweep.id == sweep_id[phy_constraint])]
    run = random.sample(runs,1)[0]

    print("SELECTED RUN:", run)
    return run
    

def load_model(run):

    global run_dir, log_dir
    run_dir = scratch_dir + run.id + "/"
    log_dir = run_dir + "logs/"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(log_dir)

    config = Struct(**run.config)
    try:
        config.phy_constraint is None
    except:
        config.phy_constraint = None

    print("LOADING WEIGHTS")
    run.file("model.h5").download(root=run_dir, replace=True)
    state_dict = torch.load(run_dir+"model.h5")
    AE = Autoencoder(config).cuda()
    AE.load_state_dict(state_dict)

    return AE

class WandbLog(BaseCallback):
    def __init__(self, name, eval_env, freq, verbose=0):
        super().__init__(verbose)
        self.name = name
        self.eval_env = eval_env
        self.freq = freq
        self.count = 0

    def _on_step(self):
        self.count += self.freq
        mean_reward, std_reward = evaluate_policy(self.model, 
                                                self.eval_env, 
                                                n_eval_episodes=5, 
                                                deterministic=True)
        wandb.log({f"Average Reward - {self.name}": mean_reward}, step=self.count)
        wandb.log({f"Std. Dev. Reward - {self.name}": std_reward}, step=self.count)

def create_env(model):
    print("CREATING ENVIRONMENTS")
    train_env = LearnedEnv(model, model.config)
    # check_env(train_env)
    train_env.reset()

    eval_env = EncodedEnv(model, model.config)
    # check_env(eval_env)
    train_ag = sb3.PPO("MlpPolicy", train_env, tensorboard_log=log_dir)

    eval_freq = int(5e3)
    train_freq = int(2e2)
    # eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir,
    #                             eval_freq=eval_freq, deterministic=True, render=False,
    #                             callback_after_eval=WandbLog(eval_env,eval_freq))
    eval_callback  = EveryNTimesteps(eval_freq,  WandbLog("Eval",eval_env,eval_freq))
    train_callback = EveryNTimesteps(train_freq, WandbLog("Train",train_env,train_freq))
    
    
    return train_ag, lambda steps: train_ag.learn(total_timesteps=steps, \
                                                    callback=[
                                                        eval_callback,
                                                        train_callback,
                                                        WandbCallback(gradient_save_freq=eval_freq)
                                                    ])


# sweep_id = {
#     "0": "bwtxt4mg", # REGULAR
#     "1": "zw8dz05t", # CONSTRAINED 1
#     "2": "2pbv8gut", # CONSTRAINED 2
#     # "3": "", # CONSTRAINED 3
#     "4": "ju9eq13d", # CONSTRAINED 4
# }
sweep_id = {
    "4": "bwtxt4mg", # REGULAR
    "1": "zw8dz05t", # CONSTRAINED 1
    "2": "2pbv8gut", # CONSTRAINED 2
    "3": "ju9eq13d", # CONSTRAINED 4
}


if __name__ == "__main__":
    import sys
    assert len(sys.argv)==2, "Usage: python phyrl.py [Physics Constraint Index]"
    phy_constraint = sys.argv[1]

    online = True if IBEX else False
    # online = True if IBEX else False

    try:
        while True:
            r = get_run(phy_constraint)
            run = wandb.init(entity="lucascamara", 
                            project="PhyRL_Agents", 
                            config=r.config)

            AE = load_model(r)
            agent, train_fn = create_env(AE)
            train_fn(int(5e4))
            # agent.save(run_dir+"final_agent")

            run.finish()
    except KeyboardInterrupt:
        run.finish()
