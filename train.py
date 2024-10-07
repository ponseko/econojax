from environment.economy import EconomyEnv
from algorithms.ppo_trainer import build_ppo_trainer, PpoTrainerParams
from util.logging import log_eval_logs_to_wandb
import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import wandb
import numpy as np
import os
import time

WANDB_PROJECT_NAME = "EconoJax"
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-w", "--wandb", action="store_true")
argument_parser.add_argument("-s", "--seed", type=int, default=42)
argument_parser.add_argument("-ps", "--population_seed", type=int, default=42)
argument_parser.add_argument("-a", "--num_agents", type=int, default=32)
argument_parser.add_argument("-e", "--num_envs", type=int, default=6)
argument_parser.add_argument("-t", "--total_timesteps", type=int, default=5e6)
argument_parser.add_argument("-r", "--num_resources", type=int, default=2)
argument_parser.add_argument("-d", "--debug", action="store_true")
argument_parser.add_argument("-l", "--load_model", type=str, default=None)
argument_parser.add_argument("-i", "--individual_policies", action="store_true")
argument_parser.add_argument("-g", "--enable_government", action="store_true")
argument_parser.add_argument("-wg", "--wandb_group", type=str, default=None)
argument_parser.add_argument("-np", "--network_size_pop", nargs="+", type=int, default=[128, 128])
argument_parser.add_argument("-ng", "--network_size_gov", nargs="+", type=int, default=[128, 128])
argument_parser.add_argument("--trade_prices", nargs="+", type=int, default=np.arange(1,11, dtype=int))
args, extra_args = argument_parser.parse_known_args()

# Convert extra_args to a dictionary. we assume that they set environment parameters.
env_parameters = {}
for i in range(0, len(extra_args), 2):
    key = extra_args[i].lstrip('--')
    # check if the value is a float or an int
    if "." in extra_args[i + 1]:
        try:
            value = float(extra_args[i + 1])
        except ValueError:
            value = extra_args[i + 1]
    else:
        try:
            value = int(extra_args[i + 1])
        except ValueError:
            value = extra_args[i + 1]
    # if its a False or True string, convert it to a boolean
    if value == "False":
        value = False
    elif value == "True":
        value = True
    env_parameters[key] = value

rng = jax.random.PRNGKey(args.population_seed) 
rng, ratio_seed, shuffle_seed = jax.random.split(rng, 3)
max_bonus_craft = 5
max_bonus_gather = 3
ratios = np.random.pareto(1, (50_000, ))
ratios = jax.random.pareto(
    ratio_seed, 
    1, 
    shape=(10000, args.num_agents, args.num_resources + 1)
).sort().mean(axis=0)
ratios = ratios / ratios.sum(axis=1, keepdims=True)
ratios = jax.random.permutation(shuffle_seed, ratios, axis=1, independent=True)
ratios = ratios + jax.random.normal(ratio_seed, ratios.shape) * 0.5 # some noise added
ratios = jnp.maximum(0, ratios)
ratios = ratios / ratios.sum(axis=1, keepdims=True)
ratios = jnp.nan_to_num(ratios, nan=0.0)
# order on the first skill, so we know the first agents are skilled at crafting
ratios = ratios[ratios[:, 0].argsort(descending=True)]
craft_skills = max_bonus_craft * ratios[:, 0]
gather_skills = max_bonus_gather * ratios[:, 1:]
print("skills\n", jnp.concatenate([craft_skills[:, None], gather_skills], axis=1))

env = EconomyEnv(
    seed=args.population_seed,
    num_population=args.num_agents,
    num_resources=args.num_resources,
    init_craft_skills=craft_skills,
    init_gather_skills=gather_skills,
    enable_government=args.enable_government,
    possible_trade_prices=args.trade_prices,
    **env_parameters
)
print("skills\n", jnp.concatenate([env.init_craft_skills[:, None], env.init_gather_skills], axis=1))

config = PpoTrainerParams(
    total_timesteps=args.total_timesteps,
    num_envs=args.num_envs,
    debug=args.debug,
    trainer_seed=args.seed,
    shared_policies=not args.individual_policies,
    num_log_episodes_after_training=3,
    network_size_pop=args.network_size_pop,
    network_size_gov=args.network_size_gov,
)

merged_config = {**config.__dict__, **env.__dict__}
merged_config = jax.tree.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, merged_config)
_, merged_config = eqx.partition(merged_config, eqx.is_array)

if args.wandb:
    wandb.init(
        project=WANDB_PROJECT_NAME,
        config=merged_config,
        group=args.wandb_group
    )

train_func = build_ppo_trainer(env, config, args.load_model)
train_func_jit = eqx.filter_jit(train_func, backend=config.backend)
out = train_func_jit()
trained_agent = out["train_state"]
metrics = out["train_metrics"]
eval_rewards = out["eval_rewards"]
eval_logs = out["eval_logs"]

# save the model
if not os.path.exists("models"):
    os.makedirs("models")
eqx.tree_serialise_leaves(f"models/ppo_{time.time()}", trained_agent)

if args.wandb:
    wandb.finish()
    log_eval_logs_to_wandb(eval_logs, args, WANDB_PROJECT_NAME, merged_config, env)
