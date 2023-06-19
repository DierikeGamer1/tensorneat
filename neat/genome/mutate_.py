"""
Mutate a genome.
The calculation method is the same as the mutation operation in NEAT-python.
See https://neat-python.readthedocs.io/en/latest/_modules/genome.html#DefaultGenome.mutate
"""
from typing import Tuple, Dict
from functools import partial

import jax
from jax import numpy as jnp
from jax import jit, Array

from .utils import fetch_random, fetch_first, I_INT, unflatten_connections
from .genome_ import add_node, delete_node_by_idx, delete_connection_by_idx, add_connection
from .graph import check_cycles


@jit
def mutate(rand_key: Array, nodes: Array, connections: Array, new_node_key: int, jit_config: Dict):
    """
    :param rand_key:
    :param nodes: (N, 5)
    :param connections: (2, N, N)
    :param new_node_key:
    :param jit_config:
    :return:
    """

    def m_add_node(rk, n, c):
        return mutate_add_node(rk, n, c, new_node_key, jit_config['bias_init_mean'], jit_config['response_init_mean'],
                               jit_config['activation_default'], jit_config['aggregation_default'])

    def m_add_connection(rk, n, c):
        return mutate_add_connection(rk, n, c, jit_config['input_idx'], jit_config['output_idx'])

    def m_delete_node(rk, n, c):
        return mutate_delete_node(rk, n, c, jit_config['input_idx'], jit_config['output_idx'])

    def m_delete_connection(rk, n, c):
        return mutate_delete_connection(rk, n, c)

    r1, r2, r3, r4, rand_key = jax.random.split(rand_key, 5)

    # structural mutations
    # mutate add node
    r = rand(r1)
    aux_nodes, aux_connections = m_add_node(r1, nodes, connections)
    nodes = jnp.where(r < jit_config['node_add_prob'], aux_nodes, nodes)
    connections = jnp.where(r < jit_config['node_add_prob'], aux_connections, connections)

    # mutate add connection
    r = rand(r2)
    aux_nodes, aux_connections = m_add_connection(r3, nodes, connections)
    nodes = jnp.where(r < jit_config['conn_add_prob'], aux_nodes, nodes)
    connections = jnp.where(r < jit_config['conn_add_prob'], aux_connections, connections)

    # mutate delete node
    r = rand(r3)
    aux_nodes, aux_connections = m_delete_node(r2, nodes, connections)
    nodes = jnp.where(r < jit_config['node_delete_prob'], aux_nodes, nodes)
    connections = jnp.where(r < jit_config['node_delete_prob'], aux_connections, connections)

    # mutate delete connection
    r = rand(r4)
    aux_nodes, aux_connections = m_delete_connection(r4, nodes, connections)
    nodes = jnp.where(r < jit_config['conn_delete_prob'], aux_nodes, nodes)
    connections = jnp.where(r < jit_config['conn_delete_prob'], aux_connections, connections)

    # value mutations
    nodes, connections = mutate_values(rand_key, nodes, connections, jit_config)

    return nodes, connections


@jit
def mutate_values(rand_key: Array, nodes: Array, cons: Array, jit_config: Dict) -> Tuple[Array, Array]:
    """
    Mutate values of nodes and connections.

    Args:
        rand_key: A random key for generating random values.
        nodes: A 2D array representing nodes.
        cons: A 3D array representing connections.
        jit_config: A dict containing configuration for jit-able functions.

    Returns:
        A tuple containing mutated nodes and connections.
    """

    k1, k2, k3, k4, k5, rand_key = jax.random.split(rand_key, num=6)
    bias_new = mutate_float_values(k1, nodes[:, 1], bias_mean, bias_std,
                                   bias_mutate_strength, bias_mutate_rate, bias_replace_rate)
    response_new = mutate_float_values(k2, nodes[:, 2], response_mean, response_std,
                                       response_mutate_strength, response_mutate_rate, response_replace_rate)
    weight_new = mutate_float_values(k3, cons[:, 2], weight_mean, weight_std,
                                     weight_mutate_strength, weight_mutate_rate, weight_replace_rate)
    act_new = mutate_int_values(k4, nodes[:, 3], act_list, act_replace_rate)
    agg_new = mutate_int_values(k5, nodes[:, 4], agg_list, agg_replace_rate)

    # mutate enabled
    r = jax.random.uniform(rand_key, cons[:, 3].shape)
    enabled_new = jnp.where(r < enabled_reverse_rate, 1 - cons[:, 3], cons[:, 3])
    enabled_new = jnp.where(~jnp.isnan(cons[:, 3]), enabled_new, jnp.nan)

    nodes = nodes.at[:, 1].set(bias_new)
    nodes = nodes.at[:, 2].set(response_new)
    nodes = nodes.at[:, 3].set(act_new)
    nodes = nodes.at[:, 4].set(agg_new)
    cons = cons.at[:, 2].set(weight_new)
    cons = cons.at[:, 3].set(enabled_new)
    return nodes, cons


@jit
def mutate_float_values(rand_key: Array, old_vals: Array, mean: float, std: float,
                        mutate_strength: float, mutate_rate: float, replace_rate: float) -> Array:
    """
    Mutate float values of a given array.

    Args:
        rand_key: A random key for generating random values.
        old_vals: A 1D array of float values to be mutated.
        mean: Mean of the values.
        std: Standard deviation of the values.
        mutate_strength: Strength of the mutation.
        mutate_rate: Rate of the mutation.
        replace_rate: Rate of the replacement.

    Returns:
        A mutated 1D array of float values.
    """
    k1, k2, k3, rand_key = jax.random.split(rand_key, num=4)
    noise = jax.random.normal(k1, old_vals.shape) * mutate_strength
    replace = jax.random.normal(k2, old_vals.shape) * std + mean
    r = jax.random.uniform(k3, old_vals.shape)
    new_vals = old_vals
    new_vals = jnp.where(r < mutate_rate, new_vals + noise, new_vals)
    new_vals = jnp.where(
        jnp.logical_and(mutate_rate < r, r < mutate_rate + replace_rate),
        replace,
        new_vals
    )
    new_vals = jnp.where(~jnp.isnan(old_vals), new_vals, jnp.nan)
    return new_vals


@jit
def mutate_int_values(rand_key: Array, old_vals: Array, val_list: Array, replace_rate: float) -> Array:
    """
    Mutate integer values (act, agg) of a given array.

    Args:
        rand_key: A random key for generating random values.
        old_vals: A 1D array of integer values to be mutated.
        val_list: List of the integer values.
        replace_rate: Rate of the replacement.

    Returns:
        A mutated 1D array of integer values.
    """
    k1, k2, rand_key = jax.random.split(rand_key, num=3)
    replace_val = jax.random.choice(k1, val_list, old_vals.shape)
    r = jax.random.uniform(k2, old_vals.shape)
    new_vals = old_vals
    new_vals = jnp.where(r < replace_rate, replace_val, new_vals)
    new_vals = jnp.where(~jnp.isnan(old_vals), new_vals, jnp.nan)
    return new_vals


@jit
def mutate_add_node(rand_key: Array, nodes: Array, cons: Array, new_node_key: int,
                    default_bias: float = 0, default_response: float = 1,
                    default_act: int = 0, default_agg: int = 0) -> Tuple[Array, Array]:
    """
    Randomly add a new node from splitting a connection.
    :param rand_key:
    :param new_node_key:
    :param nodes:
    :param cons:
    :param default_bias:
    :param default_response:
    :param default_act:
    :param default_agg:
    :return:
    """
    # randomly choose a connection
    i_key, o_key, idx = choice_connection_key(rand_key, nodes, cons)

    def nothing():  # there is no connection to split
        return nodes, cons

    def successful_add_node():
        # disable the connection
        new_nodes, new_cons = nodes, cons
        new_cons = new_cons.at[idx, 3].set(False)

        # add a new node
        new_nodes, new_cons = \
            add_node(new_nodes, new_cons, new_node_key,
                     bias=default_bias, response=default_response, act=default_act, agg=default_agg)

        # add two new connections
        w = new_cons[idx, 2]
        new_nodes, new_cons = add_connection(new_nodes, new_cons, i_key, new_node_key, weight=1, enabled=True)
        new_nodes, new_cons = add_connection(new_nodes, new_cons, new_node_key, o_key, weight=w, enabled=True)
        return new_nodes, new_cons

    # if from_idx == I_INT, that means no connection exist, do nothing
    nodes, cons = jax.lax.cond(idx == I_INT, nothing, successful_add_node)

    return nodes, cons


# TODO: Need we really need to delete a node?
@jit
def mutate_delete_node(rand_key: Array, nodes: Array, cons: Array,
                       input_keys: Array, output_keys: Array) -> Tuple[Array, Array]:
    """
    Randomly delete a node. Input and output nodes are not allowed to be deleted.
    :param rand_key:
    :param nodes:
    :param cons:
    :param input_keys:
    :param output_keys:
    :return:
    """
    # randomly choose a node
    node_key, node_idx = choice_node_key(rand_key, nodes, input_keys, output_keys,
                                         allow_input_keys=False, allow_output_keys=False)

    def nothing():
        return nodes, cons

    def successful_delete_node():
        # delete the node
        aux_nodes, aux_cons = delete_node_by_idx(nodes, cons, node_idx)

        # delete all connections
        aux_cons = jnp.where(((aux_cons[:, 0] == node_key) | (aux_cons[:, 1] == node_key))[:, jnp.newaxis],
                             jnp.nan, aux_cons)

        return aux_nodes, aux_cons

    nodes, cons = jax.lax.cond(node_idx == I_INT, nothing, successful_delete_node)

    return nodes, cons


@jit
def mutate_add_connection(rand_key: Array, nodes: Array, cons: Array,
                          input_keys: Array, output_keys: Array) -> Tuple[Array, Array]:
    """
    Randomly add a new connection. The output node is not allowed to be an input node. If in feedforward networks,
    cycles are not allowed.
    :param rand_key:
    :param nodes:
    :param cons:
    :param input_keys:
    :param output_keys:
    :return:
    """
    # randomly choose two nodes
    k1, k2 = jax.random.split(rand_key, num=2)
    i_key, from_idx = choice_node_key(k1, nodes, input_keys, output_keys,
                                      allow_input_keys=True, allow_output_keys=True)
    o_key, to_idx = choice_node_key(k2, nodes, input_keys, output_keys,
                                    allow_input_keys=False, allow_output_keys=True)

    con_idx = fetch_first((cons[:, 0] == i_key) & (cons[:, 1] == o_key))

    def successful():
        new_nodes, new_cons = add_connection(nodes, cons, i_key, o_key, weight=1, enabled=True)
        return new_nodes, new_cons

    def already_exist():
        new_cons = cons.at[con_idx, 3].set(True)
        return nodes, new_cons

    def cycle():
        return nodes, cons

    is_already_exist = con_idx != I_INT
    unflattened = unflatten_connections(nodes, cons)
    is_cycle = check_cycles(nodes, unflattened, from_idx, to_idx)

    choice = jnp.where(is_already_exist, 0, jnp.where(is_cycle, 1, 2))
    nodes, cons = jax.lax.switch(choice, [already_exist, cycle, successful])
    return nodes, cons


@jit
def mutate_delete_connection(rand_key: Array, nodes: Array, cons: Array):
    """
    Randomly delete a connection.
    :param rand_key:
    :param nodes:
    :param cons:
    :return:
    """
    # randomly choose a connection
    i_key, o_key, idx = choice_connection_key(rand_key, nodes, cons)

    def nothing():
        return nodes, cons

    def successfully_delete_connection():
        return delete_connection_by_idx(nodes, cons, idx)

    nodes, cons = jax.lax.cond(idx == I_INT, nothing, successfully_delete_connection)

    return nodes, cons


@partial(jit, static_argnames=('allow_input_keys', 'allow_output_keys'))
def choice_node_key(rand_key: Array, nodes: Array,
                    input_keys: Array, output_keys: Array,
                    allow_input_keys: bool = False, allow_output_keys: bool = False) -> Tuple[Array, Array]:
    """
    Randomly choose a node key from the given nodes. It guarantees that the chosen node not be the input or output node.
    :param rand_key:
    :param nodes:
    :param input_keys:
    :param output_keys:
    :param allow_input_keys:
    :param allow_output_keys:
    :return: return its key and position(idx)
    """

    node_keys = nodes[:, 0]
    mask = ~jnp.isnan(node_keys)

    if not allow_input_keys:
        mask = jnp.logical_and(mask, ~jnp.isin(node_keys, input_keys))

    if not allow_output_keys:
        mask = jnp.logical_and(mask, ~jnp.isin(node_keys, output_keys))

    idx = fetch_random(rand_key, mask)
    key = jnp.where(idx != I_INT, nodes[idx, 0], jnp.nan)
    return key, idx


@jit
def choice_connection_key(rand_key: Array, nodes: Array, cons: Array) -> Tuple[Array, Array, Array]:
    """
    Randomly choose a connection key from the given connections.
    :param rand_key:
    :param nodes:
    :param cons:
    :return: i_key, o_key, idx
    """

    idx = fetch_random(rand_key, ~jnp.isnan(cons[:, 0]))
    i_key = jnp.where(idx != I_INT, cons[idx, 0], jnp.nan)
    o_key = jnp.where(idx != I_INT, cons[idx, 1], jnp.nan)

    return i_key, o_key, idx


@jit
def rand(rand_key):
    return jax.random.uniform(rand_key, ())
