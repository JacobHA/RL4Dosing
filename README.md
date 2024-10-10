Welcome to the code for our NeurIPS workshop paper!
"Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics" by Josiah Kratz and Jacob Adamczyk

In this repository, you will find model code to run the dynamical system with FDEs, a Gymnasium Environment class based on the cellular dynamics with dosing, and a basic harness for hyperparameter sweeping and testing. Below is a brief summary of each file's function:

- cell_model_pop_fde_slow_sde.py: Simulating the dynamics under given model parameters and memory strength.
- cell_env.py: Gymnasium environment for using the cell dynamics as an RL environment.
- ddqn.py: Extension of SB3's DQN class that also uses the Double DQN action selection de-coupling.
- hparams.yaml: Hyperparameter ranges used for sweeping. Can be used to initialize a sweep on w&b.
- local_run.py: Used to test the finetuned hyperparameters in a local run (may take a long time)
- plot_data.ipynb: Helper scripts to visualize experiment results
- run_models.py: Run the aggregated best models for each value of mu (alpha in the code).
- sde_test.ipynb: Used to run the baseline 
- switching_at_const_frac.ipynb: Used to run the baseline
- wandb_sweep.py: Used for configuring and running sweeps 