import time
import matplotlib.pyplot as plt
import wandb
import numpy as np
import jax
import pickle
import os

def log_eval_logs_to_wandb(log, args, wandb_project_name, config, env, id=None):
    if args.wandb_group is None:
        group_name = f"eval_logs_{int(time.time())}"
    else:
        group_name = args.wandb_group
    num_envs = log["timestep"].shape[0]
    num_steps_per_episode = log["timestep"].shape[1]
    for env_id in range(num_envs):
        run = wandb.init(
            project=wandb_project_name,
            config=config,
            tags=["eval"],
            group=group_name,
        )
        for step in range(0, num_steps_per_episode, 25):
            log_step = jax.tree.map(lambda x: x[env_id, step], log)
            run.log(log_step)

        # ACTION DISTRIBUTION
        bins = np.concatenate([
            np.arange(0, env.trade_actions_total, len(env.possible_trade_prices), dtype=int),
            np.arange(env.trade_actions_total, env.action_space("population").n + 1, dtype=int),
        ], dtype=int)

        resources = config["num_resources"]
        shared_policy = config["share_policy_nets"]
        shared_values = config["share_value_nets"]
        agent_ids = config["insert_agent_ids"]
        seed = config["trainer_seed"]
        pop_seed = config["seed"]
        wandb_id = run.id
        folder = "population_actions"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the action distribution dict as a file
        name = f"{folder}/_r-{resources}_s-{shared_policy}_sv-{shared_values}_ai-{agent_ids}_seed-{seed}_popseed-{pop_seed}-{id}"
        with open(f"{name}_{wandb_id}.pkl", "wb") as f:
            pickle.dump(log["population_actions"], f)

        agent_action_dists = []
        for agent_id, agent_actions in log["population_actions"].items():
            counts, _ = np.histogram(agent_actions[env_id], bins=bins)
            agent_action_dists.append(counts)
        agent_action_dists = np.stack(agent_action_dists, axis=0)
        pairwise_diffs = np.abs(agent_action_dists[:, None] - agent_action_dists)
        total_diffs = pairwise_diffs.sum() // 2
        run.summary[f"Total action dist differences"] = total_diffs

        # Send action distribution as image to wandb:
        for agent_id, agent_actions in log["population_actions"].items():
            counts, _ = np.histogram(agent_actions[env_id], bins=bins)

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
            run.log({f"Action dist agent {agent_id}": wandb.Image(fig)}, commit=agent_id == len(log["population_actions"]) - 1)
            plt.close()

        run.finish()