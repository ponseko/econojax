from environment.economy import EconomyEnv
from algorithms.ppo_trainer import build_ppo_trainer, PpoTrainerParams
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
args = argument_parser.parse_args()

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
    num_population=args.num_agents,
    num_resources=args.num_resources,
    craft_skills=craft_skills,
    gather_skills=gather_skills,
    enable_government=args.enable_government
)

config = PpoTrainerParams(
    total_timesteps=args.total_timesteps,
    num_envs=args.num_envs,
    debug=args.debug,
    trainer_seed=args.seed,
    shared_policies=not args.individual_policies,
    num_log_episodes_after_training=3
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

def log_eval_logs_to_wandb(log):
    import time
    import matplotlib.pyplot as plt
    if args.wandb_group is None:
        group_name = f"eval_logs_{int(time.time())}"
    else:
        group_name = args.wandb_group
    num_envs = log["timestep"].shape[0]
    num_steps_per_episode = log["timestep"].shape[1]
    for env_id in range(num_envs):
        run = wandb.init(
            project=WANDB_PROJECT_NAME,
            config=merged_config,
            tags=["eval"],
            group=group_name,
        )
        for step in range(0, num_steps_per_episode, 20):
            log_step = jax.tree.map(lambda x: x[env_id, step], log)
            log_step.pop("terminal_observation")
            run.log(log_step)

        # ACTION DISTRIBUTION
        bins = np.concatenate([
            np.arange(0, env.trade_actions_total, 10, dtype=int),
            np.arange(env.trade_actions_total, env.action_space("population").n + 1, dtype=int),
        ], dtype=int)
        for agent_id, agent_actions in eval_logs["population_actions"].items():
            counts, _ = np.histogram(agent_actions[0], bins=bins)

            labels = [label for pair in zip([f"buy_{i}" for i in range(env.num_resources)], [f"sell_{i}" for i in range(env.num_resources)]) for label in pair] + [f"gather_{i}" for i in range(env.num_resources)] + ["craft"]
            if len(counts) > len(labels):
                labels.append("noop")
            action_dist = {
                label: count for label, count in zip(labels, counts)
            }
            total = sum(action_dist.values())
            action_dist = {k: v / total for k, v in action_dist.items()}
            fig = plt.figure()
            plt.bar(list(action_dist.keys()), action_dist.values())
            plt.ylabel("Percentage of actions")
            plt.ylim(0, 1)
            run.log({f"Action dist agent {agent_id}": wandb.Image(fig)}, commit=agent_id == len(eval_logs["population_actions"]) - 1)
            plt.close()
        run.finish()

if args.wandb:
    wandb.finish()
    log_eval_logs_to_wandb(eval_logs)
