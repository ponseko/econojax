# A simple training script to train a PPO agent on the EconoJax environment.
import jax
import jaxnasium as jym
from jaxnasium.algorithms import PPO

from econojax import EconoJax

if __name__ == "__main__":
    seed = jax.random.PRNGKey(42)
    env = EconoJax()
    env = jym.LogWrapper(env)
    agent = PPO(log_function="simple")
    agent = agent.train(seed, env)
