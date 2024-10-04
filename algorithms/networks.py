import jax
import equinox as eqx
from typing import List
import jax.numpy as jnp
import numpy as np

class CustomLinear(eqx.nn.Linear):
    """ eqx.nn.Linear with optional orthogonal initialization """
    def __init__(self, orth_init, orth_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if orth_init:
            weight_shape = self.weight.shape
            orth_init = jax.nn.initializers.orthogonal(orth_scale)
            self.weight = orth_init(kwargs["key"], weight_shape)
            if self.bias is not None:
                self.bias = jnp.zeros(self.bias.shape)


class ActorNetwork(eqx.Module):
    """Actor network"""

    layers: list

    def __init__(self, key, in_shape, hidden_features: List[int], num_actions, orthogonal_init=True, **kwargs):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [
            CustomLinear(orthogonal_init, np.sqrt(2), in_shape, hidden_features[0], key=keys[0])
        ] + [
            CustomLinear(orthogonal_init, np.sqrt(2), hidden_features[i], hidden_features[i+1], key=keys[i+1])
            for i in range(len(hidden_features)-1)
        ] + [
            CustomLinear(orthogonal_init, 0.01, hidden_features[-1], num_actions, key=keys[-1])
        ]

    def __call__(self, x):
        if isinstance(x, dict):
            action_mask = x["action_mask"]
            x = x["observation"]
        else: action_mask = None
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        logits = self.layers[-1](x)
        if action_mask is not None:
            logit_mask = jnp.ones_like(logits) * -1e8
            logits_mask = logit_mask * (1 - action_mask)
            logits = logits + logits_mask
        return logits
    
class ActorNetworkMultiDiscrete(eqx.Module):
    """'
    Actor network for a multidiscrete output space
    """

    layers: list
    output_heads: list

    def __init__(self, key, in_shape, hidden_features, actions_nvec, orthogonal_init=True, **kwargs):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [CustomLinear(orthogonal_init, np.sqrt(2), in_shape, hidden_features[0], key=keys[0])]
        for i, feature in enumerate(hidden_features[:-1]):
            self.layers.append(
                CustomLinear(orthogonal_init, np.sqrt(2), feature, hidden_features[i + 1], key=keys[i])
            )

        multi_discrete_heads_keys = jax.random.split(keys[-1], len(actions_nvec))
        try: actions_nvec = actions_nvec.tolist() # convert to list if numpy array
        except AttributeError: pass
        self.output_heads = [
            CustomLinear(orthogonal_init, np.sqrt(2), hidden_features[-1], action, key=multi_discrete_heads_keys[i])
            for i, action in enumerate(actions_nvec)
        ]
        if len(set(actions_nvec)) == 1:  # all output shapes are the same, vmap
            self.output_heads = jax.tree_util.tree_map(
                lambda *v: jnp.stack(v), *self.output_heads
            )
        else:
            raise NotImplementedError(
                "Different output shapes are not supported"
            )

    def __call__(self, x):
        if isinstance(x, dict):
            action_mask = x["action_mask"]
            x = x["observation"]
        else: action_mask = None

        def forward(head, x):
            return head(x)

        for layer in self.layers:
            x = jax.nn.tanh(layer(x))
        logits = jax.vmap(forward, in_axes=(0, None))(self.output_heads, x)

        if action_mask is not None:  # mask the logits
            logit_mask = jnp.ones_like(logits) * -1e8
            logit_mask = logit_mask * (1 - action_mask)
            logits = logits + logit_mask

        return logits
    
class ValueNetwork(eqx.Module):
    """
        Value (critic) network with a single output
        Used to output V when given a state
    """
    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int], orthogonal_init=True, **kwargs):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [
            CustomLinear(orthogonal_init, np.sqrt(2), in_shape, hidden_layers[0], key=keys[0])
        ] + [
            CustomLinear(orthogonal_init, np.sqrt(2), hidden_layers[i], hidden_layers[i+1], key=keys[i+1])
            for i in range(len(hidden_layers)-1)
        ] + [
            CustomLinear(orthogonal_init, 0.01, hidden_layers[-1], 1, key=keys[-1])
        ]

    def __call__(self, x):
        if isinstance(x, dict):
            x = x["observation"]
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)