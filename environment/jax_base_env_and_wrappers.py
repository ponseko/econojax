import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Union, NamedTuple
import equinox as eqx
import wandb
import numpy as np
import time

@chex.dataclass(frozen=True)
class EnvState:
    time: int


class TimeStep(NamedTuple):
    observation: chex.Array
    reward: Union[float, chex.Array]
    done: bool
    discount: Union[float, chex.Array]
    info: dict


class JaxBaseEnv(eqx.Module):
    """
    Base class for a JAX environment.
    This class inherits from eqx.Module, meaning it is a PyTree node and a dataclass.
    set params by setting the properties of the class.
    Much of the modules are inspired by the Gymnax base class.
    """

    # example_property: int = 0

    def __check_init__(self):
        """
        An equinox module that always runs on initialization.
        Can be used to check if parameters are set correctly, without overwriting __init__.
        """
        pass

    def step(
        self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array]
    ) -> Tuple[TimeStep, EnvState]:
        """Performs step transitions in the environment."""

        (obs_step, reward, terminated, truncated, info), state_step = self.step_env(
            key, state, action
        )
        obs_reset, state_reset = self.reset_env(key)

        done = jnp.any(jnp.logical_or(terminated["population"], truncated["population"]))
        
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.lax.cond(done, lambda: obs_reset, lambda: obs_step)

        info["terminal_observation"] = obs_step

        return TimeStep(obs, reward, terminated, truncated, info), state

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key)
        return obs, state

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset transition."""
        raise NotImplementedError

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array]
    ) -> Tuple[TimeStep, EnvState]:
        """Environment-specific step transition."""
        raise NotImplementedError


class JaxEnvWrapper(eqx.Module):
    _env: eqx.Module 
    
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


@chex.dataclass(frozen=True)
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    returned_episode_returns: float
    timestep: int


class LogWrapper(JaxEnvWrapper):
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key)
        num_agents = self._env.num_agents
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros(num_agents),
            returned_episode_returns=jnp.zeros(num_agents),
            timestep=0,
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float, chex.Array],
    ) -> Tuple[TimeStep, LogEnvState]:
        (obs, reward, terminated, truncated, info), env_state = self._env.step(
            key, state.env_state, action
        )
        ep_rewards = jax.tree.leaves(reward)
        ep_rewards = jnp.concatenate([ep_rewards[1], jnp.expand_dims(ep_rewards[0], axis=-1)])
        done = jnp.any(jnp.logical_or(terminated["population"], truncated["population"]))
        new_episode_return = state.episode_returns + ep_rewards
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode"] = done
        info["timestep"] = state.timestep
        return TimeStep(obs, reward, terminated, truncated, info), state