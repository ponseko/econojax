from environment.economy import EconomyEnv
from algorithms.ppo_trainer import build_ppo_trainer, PpoTrainerParams
from util.logging import log_eval_logs_to_wandb
from util.util_functions import get_pareto_skill_dists
import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import wandb
import numpy as np
import os
import time

wandb.require("core")

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
argument_parser.add_argument("-iv", "--individual_value_nets", action="store_true")
argument_parser.add_argument("-g", "--enable_government", action="store_true")
argument_parser.add_argument("-wg", "--wandb_group", type=str, default=None)
argument_parser.add_argument("-npp", "--network_size_pop_policy", nargs="+", type=int, default=[128, 128])
argument_parser.add_argument("-npv", "--network_size_pop_value", nargs="+", type=int, default=[128, 128])
argument_parser.add_argument("-ng", "--network_size_gov", nargs="+", type=int, default=[128, 128])
argument_parser.add_argument("--trade_prices", nargs="+", type=int, default=np.arange(1,11,step=2, dtype=int))
argument_parser.add_argument("--eval_runs", type=int, default=3)
argument_parser.add_argument("--rollout_length", type=int, default=150)
argument_parser.add_argument("--init_learned_skills", action="store_true")
argument_parser.add_argument("--skill_multiplier", type=float, default=0.0)
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

craft_skills, gather_skills = None, None
if args.init_learned_skills:
    craft_skills, gather_skills = get_pareto_skill_dists(args.population_seed, args.num_agents, args.num_resources)

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
    share_policy_nets=not args.individual_policies,
    share_value_nets=not args.individual_value_nets,
    num_log_episodes_after_training=args.eval_runs,
    network_size_pop_policy=args.network_size_pop_policy,
    network_size_pop_value=args.network_size_pop_value,
    network_size_gov=args.network_size_gov,
    num_steps=args.rollout_length
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

train_func, eval_func = build_ppo_trainer(env, config, args.load_model)
train_func_jit = eqx.filter_jit(train_func, backend=config.backend)
out = train_func_jit()
trained_agent = out["train_state"]
metrics = out["train_metrics"]

# save the model
if not os.path.exists("models"):
    os.makedirs("models")
eqx.tree_serialise_leaves(f"models/ppo_{time.time()}", trained_agent)

if args.wandb:
    wandb.finish()

if config.num_log_episodes_after_training > 0:
    rng = jax.random.PRNGKey(args.seed)
    rng, eval_key = jax.random.split(rng)
    eval_keys = jax.random.split(
        eval_key, config.num_log_episodes_after_training
    )
    # eval_rewards, eval_logs = jax.vmap(eval_func, in_axes=(0, None))(
    #     eval_keys, trained_agent
    # )
    # log_eval_logs_to_wandb(eval_logs, args, WANDB_PROJECT_NAME, merged_config, env)
    for i in range(len(eval_keys)):
        eval_rewards, eval_logs = eval_func(eval_key, trained_agent)
        eval_logs = jax.tree.map(lambda x: np.expand_dims(x, axis=0), eval_logs)
        log_eval_logs_to_wandb(eval_logs, args, WANDB_PROJECT_NAME, merged_config, env, id=i)
