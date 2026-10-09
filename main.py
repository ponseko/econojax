# A simple training script to train a PPO agent on the EconoJax environment.
import jax
import jaxnasium as jym
import numpy as np
from jaxnasium.algorithms import PPO
from jaxnasium.algorithms.core import mean_episode_returns

from econojax import EconoJax


def log_returns(metrics, iteration):
    returns = mean_episode_returns(metrics)
    gov_return = float(returns.pop("government"))
    pop_return = np.mean([float(r) for r in returns.values()])
    if np.isnan(pop_return):
        return  # no episode finished in this iteration
    print(
        f"iteration={int(iteration)}, population return={pop_return:.2f}, "
        f"government return={gov_return:.2f}"
    )


if __name__ == "__main__":
    seed = jax.random.PRNGKey(42)
    env = EconoJax()
    env = jym.LogWrapper(env)
    trainer = PPO(log_function=log_returns, num_envs=12)
    train_fn = jym.precompile(trainer.train, seed, env)
    agent, metrics = train_fn(seed, env)
