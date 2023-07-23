from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, Array, jit, vmap

I_INT = np.iinfo(jnp.int32).max  # infinite int
EMPTY_NODE = np.full((1, 5), jnp.nan)
EMPTY_CON = np.full((1, 4), jnp.nan)


def unflatten_conns(nodes, conns):
    """
    transform the (C, CL) connections to (CL-2, N, N)
    :return:
    """
    N = nodes.shape[0]
    CL = conns.shape[1]
    node_keys = nodes[:, 0]
    i_keys, o_keys = conns[:, 0], conns[:, 1]
    i_idxs = vmap(key_to_indices, in_axes=(0, None))(i_keys, node_keys)
    o_idxs = vmap(key_to_indices, in_axes=(0, None))(o_keys, node_keys)
    res = jnp.full((CL - 2, N, N), jnp.nan)

    # Is interesting that jax use clip when attach data in array
    # however, it will do nothing set values in an array
    # put all attributes include enable in res
    res = res.at[:, i_idxs, o_idxs].set(conns[:, 2:].T)

    return res


def key_to_indices(key, keys):
    return fetch_first(key == keys)


@jit
def fetch_first(mask, default=I_INT) -> Array:
    """
    fetch the first True index
    :param mask: array of bool
    :param default: the default value if no element satisfying the condition
    :return: the index of the first element satisfying the condition. if no element satisfying the condition, return default value
    """
    idx = jnp.argmax(mask)
    return jnp.where(mask[idx], idx, default)


@jit
def fetch_random(rand_key, mask, default=I_INT) -> Array:
    """
    similar to fetch_first, but fetch a random True index
    """
    true_cnt = jnp.sum(mask)
    cumsum = jnp.cumsum(mask)
    target = jax.random.randint(rand_key, shape=(), minval=1, maxval=true_cnt + 1)
    mask = jnp.where(true_cnt == 0, False, cumsum >= target)
    return fetch_first(mask, default)


@partial(jit, static_argnames=['reverse'])
def rank_elements(array, reverse=False):
    """
    rank the element in the array.
    if reverse is True, the rank is from small to large. default large to small
    """
    if not reverse:
        array = -array
    return jnp.argsort(jnp.argsort(array))