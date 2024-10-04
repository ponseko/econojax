import wandb
import jax.numpy as jnp
import jax
import numpy as np


def logwrapper_callback(info):
    num_envs = info["num_envs"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs
    for t in range(len(timesteps)): 
        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

def wandb_callback(info):
    if wandb.run is None:
        raise wandb.Error(
            """
                wandb logging is enabled, but wandb.run is not defined.
                Please initialize wandb before using this callback.
            """
        )
    num_envs = info["num_envs"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    if len(return_values) == 0:
        return # no episodes finished
    equalities = info["equality"][info["returned_episode"]]
    productivities = info["productivity"][info["returned_episode"]]
    coins = info["coin"][info["returned_episode"]]
    labors = info["labor"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs

    try:
        losses = info["loss_info"]
        total_loss, actor_loss, value_loss, entropy = jax.tree.map(jnp.mean, losses)
    except KeyError:
        total_loss, actor_loss, value_loss, entropy = None, None, None, None
    episode_returns_averaged = np.mean(np.array(return_values), axis=0)
    coins_averaged = np.mean(np.array(coins), axis=0)
    labors_averaged = np.mean(np.array(labors), axis=0)
    wandb.log(
        {
            "per_agent_episode_return": {
                f"{agent_id}": episode_returns_averaged[agent_id]
                for agent_id in range(len(episode_returns_averaged)-1)
            },
            "mean_population_episode_return": np.mean(episode_returns_averaged),
            "government_reward": episode_returns_averaged[-1],
            "total_episode_return_sum": np.sum(episode_returns_averaged),
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "training timestep": timesteps[-1],
            "productivity": productivities.mean(),
            "equality": equalities.mean(),
            "per_agent_coin": {
                f"{agent_id}": coins_averaged[agent_id]
                for agent_id in range(len(coins_averaged))
            },
            "per_agent_labor": {
                f"{agent_id}": labors_averaged[agent_id]
                for agent_id in range(len(labors_averaged))
            },
            "mean_population_coin": np.mean(coins_averaged),
            "mean_population_labor": np.mean(labors_averaged),
            "median_population_coin": np.median(coins_averaged),
            "median_population_labor": np.median(labors_averaged),
            "median_population_episode_return": np.median(episode_returns_averaged),
        }
    )
