import jax.numpy as jnp

def argmax_2d_array(a, dtype=None):
    """ 
        applies argmax and a custom unravel_index function. 
        Basically the same as np.unravel_index(np.argmax(a), a.shape)
        Works only on 2D arrays, but is faster compared to the original jnp/np implementation
    """
    argmax = jnp.argmax(a)
    if dtype is not None:
        return (argmax // a.shape[1]).astype(dtype), (argmax % a.shape[1]).astype(dtype)
    return argmax // a.shape[1], argmax % a.shape[1]

def get_gini(endowments):
    #From: https://github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/scenarios/utils/social_metrics.py
    # Altered for jax.numpy

    """Returns the normalized Gini index describing the distribution of endowments.

    https://en.wikipedia.org/wiki/Gini_coefficient

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized Gini index for the distribution of endowments (float). A value of 1
            indicates everything belongs to 1 agent (perfect inequality), whereas a
            value of 0 indicates all agents have equal endowments (perfect equality).

    Note:
        Uses a slightly different method depending on the number of agents. For fewer
        agents (<30), uses an exact but slow method. Switches to using a much faster
        method for more agents, where both methods produce approximately equivalent
        results.
    """
    n_agents = len(endowments)

    if n_agents < 30:  # Slower. Accurate for all n.
        diff_ij = jnp.abs(
            endowments.reshape((n_agents, 1)) - endowments.reshape((1, n_agents))
        )
        diff = jnp.sum(diff_ij)
        norm = 2 * n_agents * endowments.sum(axis=0)
        unscaled_gini = diff / (norm + 1e-10)
        gini = unscaled_gini / ((n_agents - 1) / n_agents)
        return gini

    # Much faster. Slightly overestimated for low n.
    s_endows = jnp.sort(endowments)
    return 1 - (2 / (n_agents + 1)) * jnp.sum(
        jnp.cumsum(s_endows) / (jnp.sum(s_endows) + 1e-10)
    )
