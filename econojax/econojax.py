from dataclasses import asdict, replace
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jymkit as jym
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray
from jymkit import Environment

from .util import get_gini

DEFAULT_NUM_POPULATION = 12
DEFAULT_NUM_RESOURCES = 2  # e.g. stone and wood


class EnvState(eqx.Module):
    utility: dict[str, Array]
    productivity: float
    equality: float

    inventory_coin: Array  # shape: (num_population,)
    inventory_labor: Array  # shape: (num_population,)
    inventory_resources: Array  # shape: (num_population, num_resources)

    # temp inventory for trading
    escrow_coin: Array  # shape: (num_population,)
    escrow_resources: Array  # shape: (num_population, num_resources)

    skills_craft: Array  # shape: (num_population,)
    skills_gather_resources: Array  # shape: (num_population, num_resources)

    market_orders: Array  # shape: (expiry_time, (num_resources * 2), num_population) # 2 for buy and sell
    trade_price_history: Array  # shape: (expiry_time, num_resources)

    tax_rates: Array  # shape: (len(tax_bracket_cutoffs) - 1, )
    start_year_inventory_coin: (
        Array  # shape: (num_population,) # to calculate income of the current year
    )
    income_this_period_pre_tax: Array  # shape: (num_population,)
    income_prev_period_pre_tax: Array  # shape: (num_population,)
    marginal_income: (
        Array  # shape: (num_population,) # income per earned coin after tax
    )
    net_tax_payed_prev_period: Array  # shape: (num_population,)

    timestep: int = 0


class EconoJax(Environment):
    seed: int = 0
    multi_agent = True  # Must be set to True

    num_population: int = DEFAULT_NUM_POPULATION  # Excluding the government
    num_resources: int = DEFAULT_NUM_RESOURCES
    max_steps_in_episode: int = 1000
    tax_period_length: int = 100
    enable_government: bool = True  # Enable setting tax rates
    allow_noop: bool = True
    create_info: bool = False
    insert_agent_ids: bool = False

    starting_coin: int = 15
    init_craft_skills: np.ndarray = None
    init_gather_skills: np.ndarray = None
    base_skill_development_multiplier: float = (
        0.0  # Allow skills to improve by performing actions (0 == no improvement)
    )
    max_skill_level: float = 5

    trade_expiry_time: int = 30
    max_orders_per_agent: int = 15
    possible_trade_prices: np.ndarray = eqx.field(
        converter=np.asarray,
        default_factory=lambda: np.arange(1, 11, step=2, dtype=np.int8),
    )

    coin_per_craft: int = 20  # fixed multiplier of the craft skill
    gather_labor_cost: int = 1
    craft_labor_cost: int = 1
    trade_labor_cost: int = 0.05
    craft_diff_resources_required: int = 2  # 0 = log2(num_resources) rounded down
    craft_num_resource_required: int = 2  # Requirements per resource

    tax_bracket_cutoffs: np.ndarray = eqx.field(
        converter=np.asarray,
        default_factory=lambda: np.array([0, 380.980, 755.188, np.inf]),
    )  # default: Dutch tax bracket (scaled down by 100)

    isoelastic_eta: float = 0.27
    labor_coefficient: float = 1

    @property
    def num_agents(self):
        return self.num_population + 1

    @property
    def trade_actions_per_resource(self):
        return 2 * len(self.possible_trade_prices)

    @property
    def trade_actions_total(self):
        return self.num_resources * self.trade_actions_per_resource

    def gather_resource_action_index(self, resource_index: int):  # 0-indexed
        return self.trade_actions_total + resource_index

    @property
    def craft_action_index(self):
        return self.trade_actions_total + self.num_resources

    def __post_init__(self):
        if self.craft_diff_resources_required == 0:
            diff_required = int(np.log2(self.num_resources))
            diff_required = np.clip(diff_required, 0, self.num_resources)
            self.__setattr__("craft_diff_resources_required", int(diff_required))

        key = jax.random.PRNGKey(self.seed)
        if self.init_craft_skills is None:
            init_craft_skills = jax.random.normal(key, shape=(self.num_population,)) + 1
            init_craft_skills = jnp.clip(init_craft_skills, 0, self.max_skill_level)
            self.__setattr__("init_craft_skills", init_craft_skills)
        if self.init_gather_skills is None:
            init_gather_skills = (
                jax.random.normal(key, shape=(self.num_population, self.num_resources))
                + 1
            )
            init_gather_skills = jnp.clip(init_gather_skills, 0, self.max_skill_level)
            self.__setattr__("init_gather_skills", init_gather_skills)

    def __check_init__(self):
        assert self.num_population > 0
        assert self.max_steps_in_episode > 0
        assert 1 <= self.tax_period_length < self.max_steps_in_episode
        assert len(self.init_gather_skills) == self.num_population
        assert len(self.init_craft_skills) == self.num_population
        assert self.craft_diff_resources_required >= 0
        assert self.craft_diff_resources_required <= self.num_resources
        assert self.tax_bracket_cutoffs[0] == 0, "First tax bracket should start at 0"
        assert self.tax_bracket_cutoffs[-1] == np.inf, "Last tax bracket must be np.inf"
        assert np.all(np.diff(self.tax_bracket_cutoffs) > 0), (
            "Tax brackets must be strictly increasing"
        )
        assert np.all(self.possible_trade_prices > 0), "Trade prices should be positive"
        assert np.all(np.diff(self.possible_trade_prices) > 0), (
            "Trade prices must be strictly increasing"
        )

    def reset_env(self, key: PRNGKeyArray) -> Tuple[Dict[str, Array], EnvState]:
        start_coin = jnp.ones(self.num_population, dtype=jnp.int32) * self.starting_coin
        initial_utilities = {f"p{i}": 0.0 for i in range(self.num_population)}
        initial_utilities["government"] = 0.0
        state = EnvState(
            utility=initial_utilities,
            inventory_coin=start_coin,
            inventory_labor=jnp.zeros(self.num_population),
            inventory_resources=jnp.zeros(
                (self.num_population, self.num_resources), dtype=jnp.int16
            ),
            escrow_coin=jnp.zeros(self.num_population, dtype=jnp.int32),
            escrow_resources=jnp.zeros(
                (self.num_population, self.num_resources), dtype=jnp.int16
            ),
            skills_craft=self.init_craft_skills,
            skills_gather_resources=self.init_gather_skills,
            market_orders=jnp.zeros(
                (self.trade_expiry_time, self.num_resources, 2, self.num_population),
                dtype=jnp.int8,
            ),  # 2 for buy and sell
            trade_price_history=jnp.zeros(
                (self.trade_expiry_time, self.num_resources), dtype=jnp.float16
            ),
            tax_rates=jnp.zeros(len(self.tax_bracket_cutoffs) - 1, dtype=jnp.float32),
            start_year_inventory_coin=start_coin,
            income_this_period_pre_tax=jnp.zeros(self.num_population, dtype=jnp.int32),
            income_prev_period_pre_tax=jnp.zeros(self.num_population, dtype=jnp.int32),
            marginal_income=jnp.ones(self.num_population, dtype=jnp.float32),
            net_tax_payed_prev_period=jnp.zeros(self.num_population, dtype=jnp.int32),
            timestep=0,
            productivity=0.0,
            equality=0.0,
        )
        state = self.calculate_utilities(state)
        observations = self.get_observations_and_action_masks(state)
        return observations, state

    def step_env(
        self, rng: PRNGKeyArray, state: EnvState, actions: Dict[str, Array]
    ) -> Tuple[jym.TimeStep, EnvState]:
        # actions dict is sorted [see](https://github.com/jax-ml/jax/pull/26069)
        # we re-sort the actions dict to match the action_space and split
        # into {"population": ..., "government": ...}
        population_actions = {
            key: actions[key] for key in self.action_space.keys() if key != "government"
        }
        population_actions = jnp.stack(population_actions.values())
        actions = {
            "population": population_actions,
            "government": actions["government"],
        }

        new_state = self.update_state(state, actions, rng)
        observations = self.get_observations_and_action_masks(new_state)
        reward = self.get_rewards(state, new_state)
        terminated, truncated = self.get_terminated_truncated(new_state)
        info = self.get_info(new_state, actions)

        return jym.TimeStep(
            observations, reward, terminated, truncated, info
        ), new_state

    def get_observations_and_action_masks(self, state: EnvState) -> Dict[str, Array]:
        obs = self.get_observations(state)
        action_masks = self.get_action_masks(state)
        observations = jax.tree.map(
            lambda o, m: jym.AgentObservation(o, m), obs, action_masks
        )
        return observations

    def get_observations(self, state: EnvState) -> Dict[str, Array]:
        """
        Should take in a state and return an observation array of shape (num_population, num_features)
        Also returns the action mask of shape (num_population, num_actions)
        """

        is_tax_day = (
            state.timestep % self.tax_period_length == 0
            if self.enable_government
            else 0
        )

        ### Market info
        # state.market_orders = of shape (expiry_time, num_resources, 2, num_population)
        # for each resource we need the highest buy per agent and lowest sell per agent
        highest_order_per_resource_per_agent = jnp.max(state.market_orders, axis=0)
        lowest_order_per_resource_per_agent = jnp.min(state.market_orders, axis=0)
        highest_buy_order_per_resource_per_agent = highest_order_per_resource_per_agent[
            :, 0, :
        ]
        lowest_sell_order_per_resource_per_agent = lowest_order_per_resource_per_agent[
            :, 1, :
        ]
        highest_buy_order_per_resource = jnp.max(
            highest_buy_order_per_resource_per_agent, axis=1
        )
        lowest_sell_order_per_resource = jnp.min(
            lowest_sell_order_per_resource_per_agent, axis=1
        )
        average_prices = jnp.nan_to_num(
            jnp.mean(
                state.trade_price_history, axis=0, where=state.trade_price_history != 0
            ),
            nan=0,
        )

        private_observations = [
            state.inventory_coin / 1000,
            state.inventory_resources / 10,
            state.inventory_labor / 100,
            state.escrow_coin / 1000,
            state.escrow_resources,
            state.skills_craft,
            state.skills_gather_resources,
            highest_buy_order_per_resource_per_agent.T,
            lowest_sell_order_per_resource_per_agent.T,
            state.marginal_income,
            state.income_this_period_pre_tax / 1000,
            state.income_prev_period_pre_tax / 1000,
        ]
        if self.insert_agent_ids:
            agent_ids = np.arange(self.num_population)
            binary_agent_ids = (
                (
                    agent_ids[:, None]
                    & (1 << np.arange(self.num_population.bit_length()))
                )
                > 0
            ).astype(int)[:, ::-1]
            private_observations.append(binary_agent_ids)
        private_observations = jnp.column_stack(private_observations)

        global_observations = jnp.hstack(
            [
                # state.timestep,
                state.tax_rates,
                highest_buy_order_per_resource,
                lowest_sell_order_per_resource,
                average_prices,
                is_tax_day,
            ]
        )
        global_observations = jnp.broadcast_to(
            global_observations, (self.num_population, global_observations.shape[0])
        )

        observations_population = jnp.hstack(
            [private_observations, global_observations]
        )

        # the number of agents that are in each of the brackets:
        highest_brackets_per_agent = (
            jnp.digitize(state.income_this_period_pre_tax, self.tax_bracket_cutoffs) - 1
        )
        count_per_bracket = jnp.bincount(
            highest_brackets_per_agent, length=len(self.tax_bracket_cutoffs) - 1
        )

        observation_government = jnp.concatenate(
            [
                jnp.array(
                    [
                        is_tax_day,
                        state.inventory_coin.mean() / 1000,
                        state.inventory_coin.std() / 1000,
                        jnp.median(state.inventory_coin) / 1000,
                        state.start_year_inventory_coin.mean() / 1000,
                        state.start_year_inventory_coin.std() / 1000,
                        jnp.median(state.start_year_inventory_coin) / 1000,
                        state.income_this_period_pre_tax.mean() / 1000,
                        state.income_this_period_pre_tax.std() / 1000,
                        jnp.median(state.income_this_period_pre_tax) / 1000,
                    ]
                ),
                count_per_bracket / self.num_population,
                state.tax_rates,
                average_prices / 10,
            ]
        ).flatten()

        observations = {
            f"p{i}": observations_population[i] for i in range(self.num_population)
        }
        observations["government"] = observation_government
        return observations

    def get_action_masks(self, state: EnvState) -> Array:
        # For convinience, trade actions will be the first actions, this is helpful in the trade_action_processing function
        coin_inventory = state.inventory_coin
        resources_inventory = state.inventory_resources

        ### trade
        # prices = np.arange(self.trade_price_floor, self.trade_price_ceiling + 1)
        prices = self.possible_trade_prices
        num_trade_actions = int(self.trade_actions_per_resource / 2)
        # Buy orders (need enough coin)
        buy_resource_masks = (coin_inventory[:, None] >= prices).astype(jnp.bool)
        buy_resource_masks = jnp.expand_dims(buy_resource_masks, axis=1).repeat(
            self.num_resources, axis=1
        )
        # Sell orders (need at least 1 resource)
        sell_resource_masks = (
            (resources_inventory >= 1)
            .astype(jnp.bool)[:, :, None]
            .repeat(num_trade_actions, axis=-1)
        )
        trade_masks = jnp.concatenate(
            [buy_resource_masks, sell_resource_masks], axis=-1
        ).reshape(self.num_population, -1)

        # agent can't trade if they reached the max number of orders
        num_orders_per_agent = jnp.count_nonzero(state.market_orders, axis=(0, 1, 2))
        orders_remaining = (num_orders_per_agent <= self.max_orders_per_agent)[:, None]
        trade_masks = trade_masks * orders_remaining

        ### gather: always available
        gather_masks = jnp.ones(
            (self.num_population, self.num_resources), dtype=jnp.int16
        )

        ### craft
        # craft_masks = (resources_inventory >= self.craft_num_resource_required).all(axis=1)[:, None].astype(jnp.bool)
        craft_masks = (
            (resources_inventory >= self.craft_num_resource_required).sum(axis=1)
            >= self.craft_diff_resources_required
        )[:, None].astype(jnp.bool)

        population_masks = jnp.concatenate(
            [trade_masks, gather_masks, craft_masks], axis=1
        )

        if self.allow_noop:
            do_nothing_masks = jnp.ones((self.num_population, 1), dtype=jnp.bool)
            population_masks = jnp.concatenate(
                [population_masks, do_nothing_masks], axis=1
            )

        # Gov actions: won't have any effect when not a tax day
        gov_action_space_nvec = self.action_space["government"].nvec
        government_masks = jnp.ones(
            (len(gov_action_space_nvec), gov_action_space_nvec[0]), dtype=jnp.bool
        )

        action_masks = {
            f"p{i}": population_masks[i] for i in range(self.num_population)
        }
        action_masks["government"] = government_masks
        return action_masks

    def get_rewards(self, old_state: EnvState, new_state: EnvState) -> Array:
        """Returns the difference in utility for a timestep as the reward"""
        return jax.tree.map(lambda x, y: y - x, old_state.utility, new_state.utility)

    def get_terminated_truncated(self, state: EnvState) -> Tuple[bool, bool]:
        terminated = False  # no termination
        truncated = state.timestep >= self.max_steps_in_episode
        return terminated, truncated

    def get_info(self, state: EnvState, actions) -> Dict[str, Array]:
        if not self.create_info:
            return {
                "coin": state.inventory_coin,
                "labor": state.inventory_labor,
                "productivity": state.productivity,
                "equality": state.equality,
            }
        state_dict = asdict(state)
        state_dict.update({"population_actions": actions["population"]})
        state_dict.update({"government_actions": actions["government"]})
        state_dict.update({"population_utility": state.utility["population"]})
        state_dict.update({"government_utility": state.utility["government"]})
        info_keys = [
            "inventory_coin",
            "inventory_labor",
            "skills_craft",
            "population_utility",
            "population_actions",
            "timestep",
            "tax_rates",
            "trade_price_history",
            "productivity",
            "equality",
        ]
        info = {}
        for key in info_keys:
            if state_dict[key].shape == (self.num_population,):
                info[key] = {i: state_dict[key][i] for i in range(self.num_population)}
            elif state_dict[key].shape == (self.num_population, self.num_resources):
                info[key] = {}
                for r in range(self.num_resources):
                    info[key][r] = {
                        i: state_dict[key][i, r] for i in range(self.num_population)
                    }
            elif key == "government_actions" or key == "tax_rates":
                info[key] = {i: state_dict[key][i] for i in range(len(state_dict[key]))}
            elif key == "trade_price_history":
                info[key] = jnp.nan_to_num(
                    jnp.mean(state_dict[key], axis=0, where=state_dict[key] != 0), nan=0
                )
                info[key] = {i: info[key][i] for i in range(len(info[key]))}
            else:
                info[key] = state_dict[key]
        return info

    def update_state(
        self, state: EnvState, action: Dict[str, Array], rng: PRNGKeyArray
    ) -> EnvState:
        gather_key, trade_key = jax.random.split(rng, 2)

        # POPULATION
        population_actions = action[
            "population"
        ]  # an array with a discrete action for each agent
        state = self.component_gather_and_craft(state, population_actions, gather_key)
        state = self.component_trading(state, population_actions, trade_key)

        # GOVERNMENT
        # NOTE: component taxation also updates parts of the observation that are used, even when government is disabled
        government_action = action[
            "government"
        ]  # an array with a multi-discrete action for the government
        state = self.component_taxation(state, government_action)

        # Calculate utilities
        state = self.calculate_utilities(state)

        return replace(
            state,
            timestep=state.timestep + 1,
        )

    def calculate_utilities(self, state: EnvState) -> Dict[str, Array]:
        """
        Utility functions per agent, dictating the rewards.
        Rewards are the difference in utility per timestep.
        These utility functions follow that of the AI-economist:
        https://github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/scenarios/utils/rewards.py
        """
        agent_total_coin = state.inventory_coin + state.escrow_coin

        # Population utility: isoelastic utility minus labor
        # https://en.wikipedia.org/wiki/Isoelastic_utility
        util_consumption = (agent_total_coin ** (1 - self.isoelastic_eta) - 1) / (
            1 - self.isoelastic_eta
        )
        util_labor = state.inventory_labor * self.labor_coefficient
        util_population = util_consumption - util_labor

        # Government utility: productivity * equality
        EQUALITY_WEIGHT = 1.0  # 0 (U = prod) and 1 (U = prod * eq)
        productivity = jnp.sum(agent_total_coin) / self.num_population
        equality = 1 - get_gini(agent_total_coin)
        equality_weighted = EQUALITY_WEIGHT * equality + (1 - EQUALITY_WEIGHT)
        util_government = equality_weighted * productivity

        utilities = {
            f"p{i}": util_population[i] for i in range(self.num_population)
        } | {
            "government": util_government,
        }

        return replace(
            state,
            utility=utilities,
            productivity=productivity,
            equality=equality,
        )

    def component_gather_and_craft(
        self, state: EnvState, agent_actions: Array, key: PRNGKeyArray
    ) -> EnvState:
        gather_resource_action_indices = np.arange(
            self.num_resources
        ) + self.gather_resource_action_index(0)

        key, luck_key = jax.random.split(key)
        skill_success = jax.random.uniform(luck_key, (self.num_population,), maxval=1.1)
        labor_inventories = state.inventory_labor
        resource_inventories = state.inventory_resources
        coin_inventories = state.inventory_coin

        gather_actions = (
            agent_actions == gather_resource_action_indices[:, None]
        ).T  # (num_population, num_resources)
        # gather_bonus = gather_actions * (jnp.floor(state.skills_gather_resources)).astype(jnp.int16)
        gather_bonus = gather_actions * (
            jnp.floor(state.skills_gather_resources + skill_success[:, None])
        ).astype(jnp.int16)
        resource_inventories += (gather_bonus).astype(jnp.int16)
        labor_inventories += jnp.any(gather_actions, axis=1) * self.gather_labor_cost

        craft_actions = agent_actions == self.craft_action_index
        craft_actions = craft_actions  # & (resource_inventories >= self.craft_num_resource_required).all(axis=1)
        craft_gains = craft_actions * (self.coin_per_craft * state.skills_craft).astype(
            jnp.int32
        )
        coin_inventories += craft_gains.astype(jnp.int32)

        # Crafting:
        can_craft = (resource_inventories >= self.craft_num_resource_required).sum(
            axis=1
        ) >= self.craft_diff_resources_required
        will_craft = craft_actions * can_craft

        # Fixed arrays:
        highest_resource_indices = jnp.argsort(
            resource_inventories, axis=1, descending=True
        )
        resources_to_craft_with = jnp.zeros(resource_inventories.shape, dtype=bool)
        resources_to_craft_with = resources_to_craft_with.at[
            np.arange(self.num_population)[:, None],
            highest_resource_indices[:, : self.craft_diff_resources_required],
        ].set(True)
        resource_changes = (
            resources_to_craft_with
            * self.craft_num_resource_required
            * will_craft[:, None]
        )

        resource_inventories -= resource_changes
        # resource_inventories -= (craft_actions * self.craft_num_resource_required)[:, None].astype(jnp.int16)
        labor_inventories += will_craft * self.craft_labor_cost

        # Skill development
        if self.base_skill_development_multiplier > 0:
            sched = optax.exponential_decay(
                init_value=self.base_skill_development_multiplier,
                transition_steps=5,
                decay_rate=0.005,
            )  # learning with high skill is harder
            gather_multiplier = sched(state.skills_gather_resources)
            craft_multiplier = sched(state.skills_craft)
            gather_skill_development = (gather_actions * gather_multiplier) + 1
            skills_gather_resources = jnp.minimum(
                state.skills_gather_resources * gather_skill_development,
                self.max_skill_level,
            )
            craft_skill_development = (will_craft * craft_multiplier) + 1
            skills_craft = jnp.minimum(
                state.skills_craft * craft_skill_development, self.max_skill_level
            )
            state = replace(
                state,
                skills_gather_resources=skills_gather_resources,
                skills_craft=skills_craft,
            )

        return replace(
            state,
            inventory_coin=coin_inventories,
            inventory_labor=labor_inventories,
            inventory_resources=resource_inventories,
        )

    def component_trading(
        self, state: EnvState, actions: Array, key: PRNGKeyArray
    ) -> EnvState:
        trade_time_index = state.timestep % self.trade_expiry_time
        # prices = np.arange(self.trade_price_floor, self.trade_price_ceiling + 1, dtype=jnp.int16)
        prices = self.possible_trade_prices
        num_trade_actions = self.trade_actions_total

        coin_inventory = state.inventory_coin
        coin_escrow = state.escrow_coin
        resource_inventory = state.inventory_resources
        resource_escrow = state.escrow_resources
        labor_inventory = state.inventory_labor
        market_orders = (
            state.market_orders
        )  # (expiry_time, num_resources, 2, num_population)
        trade_price_history = state.trade_price_history

        ### Expire previous bids/asks: return escrow to inventory
        expired_orders = market_orders[trade_time_index]
        expired_buys = expired_orders[:, 0, :]
        expired_sells = expired_orders[:, 1, :]
        coin_inventory += jnp.sum(expired_buys, axis=0, dtype=jnp.int32)
        coin_escrow -= jnp.sum(expired_buys, axis=0, dtype=jnp.int32)
        resource_inventory += expired_sells.astype(jnp.bool).T
        resource_escrow -= expired_sells.astype(jnp.bool).T

        ### Process actions (new bids/asks)
        # create a one-hot encoded matrix from the actions array (the first num_actions in actions are trade actions)
        one_hot_market_actions = jax.nn.one_hot(
            actions, num_trade_actions, dtype=jnp.int8
        )  # (num_population, num_trade_actions)
        one_hot_market_actions *= jnp.tile(
            prices, num_trade_actions // len(prices)
        )  # correct for pricing
        market_actions = one_hot_market_actions.reshape(
            self.num_population, self.num_resources, 2, -1
        ).sum(axis=-1, dtype=jnp.int8)
        market_actions = jnp.moveaxis(market_actions, 0, -1).astype(
            jnp.int8
        )  # (num_resources, 2, num_population) (these are the action of this timestep)

        # update market state:
        market_orders = market_orders.at[trade_time_index].set(market_actions)

        # update inventories
        agents_spend_on_buys = jnp.sum(
            market_actions[:, 0, :], axis=0, dtype=jnp.int32
        )  # (num_population,)
        agents_that_sell_per_resource = jnp.array(
            market_actions[:, 1, :], dtype=jnp.bool
        )  # (num_resources, num_population)
        agents_that_traded = jnp.any(market_actions, axis=(0, 1))  # (num_population,)

        coin_inventory -= agents_spend_on_buys
        coin_escrow += agents_spend_on_buys
        resource_inventory -= agents_that_sell_per_resource.T
        resource_escrow += agents_that_sell_per_resource.T
        labor_inventory += agents_that_traded * self.trade_labor_cost

        # Make sure the oldest orders are the first in the array (so that they are prioritized):
        market_orders_rolled = jnp.roll(market_orders, -(trade_time_index + 1), axis=0)

        def process_resource_orders(resource_orders: Array):
            reordered_orders = jnp.squeeze(resource_orders, axis=1)
            # randomize agent order such that final tie breaker is random
            random_agent_order = jax.random.permutation(
                key, np.arange(self.num_population)
            )
            reordered_orders = reordered_orders[:, :, random_agent_order]
            reordered_orders = jnp.moveaxis(reordered_orders, 1, 0)
            reordered_orders = reordered_orders.reshape(2, -1)
            agent_ids = jnp.tile(random_agent_order, self.trade_expiry_time)
            trade_times = jnp.repeat(
                np.arange(self.trade_expiry_time), self.num_population
            )

            buy_orders = reordered_orders[0]
            sell_orders = reordered_orders[1]
            order_of_buys = jnp.argsort(buy_orders, descending=True)  # [::-1]
            order_of_sells = jnp.argsort(
                jnp.where(sell_orders == 0, jnp.inf, sell_orders)
            )
            buy_orders = buy_orders[order_of_buys]
            sell_orders = sell_orders[order_of_sells]
            buy_agent_ids = agent_ids[order_of_buys]
            sell_agent_ids = agent_ids[order_of_sells]
            buy_trade_times = trade_times[order_of_buys]
            sell_trade_times = trade_times[order_of_sells]

            valid_trades = (
                (buy_orders >= sell_orders) & (buy_orders != 0) & (sell_orders != 0)
            )
            trade_prices = (
                jax.lax.select(
                    sell_trade_times > buy_trade_times, sell_orders, buy_orders
                )
            ) * valid_trades
            buyers_discounts = (buy_orders - trade_prices) * valid_trades
            num_resources_traded = valid_trades.astype(jnp.int16)

            # where num_resources_trades == 0, the trade did not occur
            # set buy_agent_ids and sell_agent_ids to self.num_population + 2 (out of bounds)
            # JAX will ignore these indices
            buy_agent_ids += (self.num_population * 2) * ~valid_trades
            sell_agent_ids += (self.num_population * 2) * ~valid_trades

            coin_inventory_changes = jnp.zeros(self.num_population, dtype=jnp.int32)
            coin_escrow_changes = jnp.zeros(self.num_population, dtype=jnp.int32)
            resource_inventory_changes = jnp.zeros(self.num_population, dtype=jnp.int16)
            resource_escrow_changes = jnp.zeros(self.num_population, dtype=jnp.int16)

            coin_inventory_changes = coin_inventory_changes.at[sell_agent_ids].add(
                trade_prices
            )
            coin_inventory_changes = coin_inventory_changes.at[buy_agent_ids].add(
                buyers_discounts
            )
            coin_escrow_changes = coin_escrow_changes.at[buy_agent_ids].add(
                -(trade_prices + buyers_discounts)
            )
            resource_inventory_changes = resource_inventory_changes.at[
                buy_agent_ids
            ].add(num_resources_traded)
            resource_escrow_changes = resource_escrow_changes.at[sell_agent_ids].add(
                -num_resources_traded
            )

            inventory_changes = jnp.stack(
                [
                    coin_inventory_changes,
                    coin_escrow_changes,
                    resource_inventory_changes,
                    resource_escrow_changes,
                ],
                axis=1,
            )

            # Price history
            avg_trade_price_this_step = jnp.nan_to_num(
                jnp.mean(trade_prices, where=trade_prices != 0), nan=0
            )

            # update the market itself, set occured trades to 0
            resource_orders = resource_orders.at[
                buy_trade_times, 0, 0, buy_agent_ids
            ].set(0)
            resource_orders = resource_orders.at[
                sell_trade_times, 0, 1, sell_agent_ids
            ].set(0)

            return (inventory_changes, avg_trade_price_this_step, resource_orders)

        per_resource_orders = jnp.split(
            market_orders_rolled, self.num_resources, axis=1
        )
        orders_out = jax.tree.map(process_resource_orders, per_resource_orders)

        inventory_changes = jnp.stack([x[0] for x in orders_out], axis=1)
        total_coin_inventory_changes = inventory_changes[:, :, 0].sum(axis=1)
        total_coin_escrow_changes = inventory_changes[:, :, 1].sum(axis=1)
        resource_inventory_changes = inventory_changes[:, :, 2].astype(jnp.int16)
        resource_escrow_changes = inventory_changes[:, :, 3].astype(jnp.int16)
        avg_trade_price_this_step = jnp.array(
            [x[1] for x in orders_out], dtype=jnp.float16
        )
        market_orders_rolled = jnp.concatenate([x[2] for x in orders_out], axis=1)

        coin_inventory += total_coin_inventory_changes
        coin_escrow += total_coin_escrow_changes
        resource_inventory += resource_inventory_changes
        resource_escrow += resource_escrow_changes
        trade_price_history = trade_price_history.at[trade_time_index].set(
            avg_trade_price_this_step
        )

        # roll back to original order
        market_orders = jnp.roll(market_orders_rolled, (trade_time_index + 1), axis=0)

        return replace(
            state,
            inventory_coin=coin_inventory,
            escrow_coin=coin_escrow,
            inventory_resources=resource_inventory,
            escrow_resources=resource_escrow,
            inventory_labor=labor_inventory,
            market_orders=market_orders,
            trade_price_history=trade_price_history,
        )

    def component_taxation(self, state: EnvState, actions: Array) -> EnvState:
        def process_tax_day(state: EnvState):
            inventory_coin = state.inventory_coin
            year_income_per_agent = income_prev_period_pre_tax
            tax_rates = state.tax_rates
            tax_bracket_cutoffs = self.tax_bracket_cutoffs

            income_in_tax_bracket = jnp.clip(
                year_income_per_agent[:, None] - tax_bracket_cutoffs[:-1],
                0,
                tax_bracket_cutoffs[1:] - tax_bracket_cutoffs[:-1],
            )
            tax_in_brackets_per_agent = income_in_tax_bracket * tax_rates
            total_tax_due_per_agent = tax_in_brackets_per_agent.sum(axis=-1)
            # can't pay more than you have in inventory (escrow is not considered)
            total_tax_due_per_agent = jnp.minimum(
                total_tax_due_per_agent, inventory_coin
            ).astype(jnp.int32)
            total_tax_due = total_tax_due_per_agent.sum()
            taxes_to_distribute = (total_tax_due // self.num_population).astype(
                jnp.int32
            )  # Uniform distribution of taxes

            # Collect taxes and redistribute
            inventory_coin -= total_tax_due_per_agent
            inventory_coin += taxes_to_distribute
            net_tax_payed = total_tax_due_per_agent - taxes_to_distribute

            # Now set the new tax rates according to the actions
            # actions is an array in [0, 20] with an element per bracket. e.g. action == 3: 15% (3 * 5%)
            new_tax_rates = jnp.array(actions, dtype=jnp.float32) * 0.05

            # Update the state along with the start_year_inventory_coin
            state = replace(
                state,
                inventory_coin=inventory_coin,
                tax_rates=new_tax_rates,
                net_tax_payed_prev_period=net_tax_payed,
            )

            return state

        is_tax_day = (
            state.timestep % self.tax_period_length == 0
            if self.enable_government
            else False
        )
        income_prev_period_pre_tax = jax.lax.select(
            is_tax_day,
            state.inventory_coin + state.escrow_coin - state.start_year_inventory_coin,
            state.income_prev_period_pre_tax,
        )
        start_year_inventory_coin = jax.lax.select(
            is_tax_day,
            state.inventory_coin + state.escrow_coin,
            state.start_year_inventory_coin,
        )
        income_this_period_pre_tax = (
            state.inventory_coin + state.escrow_coin - start_year_inventory_coin
        )

        state = jax.lax.cond(is_tax_day, process_tax_day, lambda state: state, state)

        # Marginal incomes
        highest_bracket_per_agent = (
            jnp.digitize(income_this_period_pre_tax, self.tax_bracket_cutoffs) - 1
        )
        applicable_tax_rates = state.tax_rates[highest_bracket_per_agent]
        return replace(
            state,
            marginal_income=1 - applicable_tax_rates,
            income_this_period_pre_tax=income_this_period_pre_tax,
            income_prev_period_pre_tax=income_prev_period_pre_tax,
            start_year_inventory_coin=start_year_inventory_coin,
        )

    @property
    def observation_space(self):
        obs, _ = self.reset(jax.random.PRNGKey(0))
        return {
            key: jym.Box(
                low=jnp.full(obs[key].observation.shape[0], 0),
                high=jnp.full(obs[key].observation.shape[0], 1000),
                shape=obs[key].observation.shape,
                dtype=jnp.float32,
            )
            for key in obs.keys()
        }

    @property
    def action_space(self):
        population_num_actions = 1 if self.allow_noop else 0
        population_num_actions += self.num_resources  # gather
        population_num_actions += 1  # craft
        population_num_actions += self.trade_actions_total  # trade (buy and sell)
        # for convenience, in a discrete action space, we assume the trade actions
        # to be the first (2 * self.trade_actions_total) actions
        population_action_space = jym.Discrete(population_num_actions)

        government_num_actions_per_bracket = 21  # every 5% up to 100% (incl. 0%)
        government_num_brackets = len(self.tax_bracket_cutoffs) - 1
        government_actions = np.full(
            government_num_brackets, government_num_actions_per_bracket
        )
        government_action_space = jym.MultiDiscrete(government_actions)
        return {
            f"p{i}": population_action_space for i in range(self.num_population)
        } | {"government": government_action_space}
