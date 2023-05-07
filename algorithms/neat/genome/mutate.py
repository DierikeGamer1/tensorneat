from typing import Tuple
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax import jit, vmap, Array

from .utils import fetch_random, fetch_first, I_INT
from .genome import add_node, add_connection_by_idx, delete_node_by_idx, delete_connection_by_idx
from .graph import check_cycles
from .activations import act_name2key
from .aggregations import agg_name2key


def create_mutate_function(N, config, batch: bool):
    """
    create mutate function for different situations
    :param N:
    :param config:
    :param batch: mutate for population or not
    :return:
    """
    num_inputs = config.basic.num_inputs
    num_outputs = config.basic.num_outputs
    input_idx = np.arange(num_inputs)
    output_idx = np.arange(num_inputs, num_inputs + num_outputs)

    bias = config.neat.gene.bias
    bias_default = bias.init_mean
    bias_mean = bias.init_mean
    bias_std = bias.init_stdev
    bias_mutate_strength = bias.mutate_power
    bias_mutate_rate = bias.mutate_rate
    bias_replace_rate = bias.replace_rate

    response = config.neat.gene.response
    response_default = response.init_mean
    response_mean = response.init_mean
    response_std = response.init_stdev
    response_mutate_strength = response.mutate_power
    response_mutate_rate = response.mutate_rate
    response_replace_rate = response.replace_rate

    weight = config.neat.gene.weight
    weight_mean = weight.init_mean
    weight_std = weight.init_stdev
    weight_mutate_strength = weight.mutate_power
    weight_mutate_rate = weight.mutate_rate
    weight_replace_rate = weight.replace_rate

    activation = config.neat.gene.activation
    act_default = act_name2key[activation.default]
    act_list = np.array([act_name2key[name] for name in activation.options])
    act_replace_rate = activation.mutate_rate

    aggregation = config.neat.gene.aggregation
    agg_default = agg_name2key[aggregation.default]
    agg_list = np.array([agg_name2key[name] for name in aggregation.options])
    agg_replace_rate = aggregation.mutate_rate

    enabled = config.neat.gene.enabled
    enabled_reverse_rate = enabled.mutate_rate

    genome = config.neat.genome
    add_node_rate = genome.node_add_prob
    delete_node_rate = genome.node_delete_prob
    add_connection_rate = genome.conn_add_prob
    delete_connection_rate = genome.conn_delete_prob
    single_structure_mutate = genome.single_structural_mutation

    def mutate_with_args(rand_key, nodes, connections, new_node_key):
        return mutate(rand_key, nodes, connections, new_node_key, input_idx, output_idx,
                      bias_default, bias_mean, bias_std, bias_mutate_strength, bias_mutate_rate,
                      bias_replace_rate, response_default, response_mean, response_std,
                      response_mutate_strength, response_mutate_rate, response_replace_rate,
                      weight_mean, weight_std, weight_mutate_strength, weight_mutate_rate,
                      weight_replace_rate, act_default, act_list, act_replace_rate,
                      agg_default, agg_list, agg_replace_rate, enabled_reverse_rate,
                      add_node_rate, delete_node_rate, add_connection_rate, delete_connection_rate,
                      single_structure_mutate)

    if not batch:
        rand_key_lower = jnp.zeros((2, ), dtype=jnp.uint32)
        nodes_lower = jnp.zeros((N, 5))
        connections_lower = jnp.zeros((2, N, N))
        new_node_key_lower = jnp.zeros((), dtype=jnp.int32)
        return jit(mutate_with_args).lower(rand_key_lower, nodes_lower, connections_lower, new_node_key_lower).compile()
    else:
        pop_size = config.neat.population.pop_size
        rand_key_lower = jnp.zeros((pop_size, 2), dtype=jnp.uint32)
        nodes_lower = jnp.zeros((pop_size, N, 5))
        connections_lower = jnp.zeros((pop_size, 2, N, N))
        new_node_key_lower = jnp.zeros((pop_size, ), dtype=jnp.int32)
        batched_mutate_func = jit(vmap(mutate_with_args)).lower(rand_key_lower, nodes_lower,
                                                                connections_lower, new_node_key_lower).compile()

        return batched_mutate_func


# @partial(jit, static_argnames=["single_structure_mutate"])
def mutate(rand_key: Array,
           nodes: Array,
           connections: Array,
           new_node_key: int,
           input_idx: Array,
           output_idx: Array,
           bias_default: float = 0,
           bias_mean: float = 0,
           bias_std: float = 1,
           bias_mutate_strength: float = 0.5,
           bias_mutate_rate: float = 0.7,
           bias_replace_rate: float = 0.1,
           response_default: float = 1,
           response_mean: float = 1.,
           response_std: float = 0.,
           response_mutate_strength: float = 0.,
           response_mutate_rate: float = 0.,
           response_replace_rate: float = 0.,
           weight_mean: float = 0.,
           weight_std: float = 1.,
           weight_mutate_strength: float = 0.5,
           weight_mutate_rate: float = 0.7,
           weight_replace_rate: float = 0.1,
           act_default: int = 0,
           act_list: Array = None,
           act_replace_rate: float = 0.1,
           agg_default: int = 0,
           agg_list: Array = None,
           agg_replace_rate: float = 0.1,
           enabled_reverse_rate: float = 0.1,
           add_node_rate: float = 0.2,
           delete_node_rate: float = 0.2,
           add_connection_rate: float = 0.4,
           delete_connection_rate: float = 0.4,
           single_structure_mutate: bool = True):
    """
    :param output_idx:
    :param input_idx:
    :param agg_default:
    :param act_default:
    :param response_default:
    :param bias_default:
    :param rand_key:
    :param nodes: (N, 5)
    :param connections: (2, N, N)
    :param new_node_key:
    :param bias_mean:
    :param bias_std:
    :param bias_mutate_strength:
    :param bias_mutate_rate:
    :param bias_replace_rate:
    :param response_mean:
    :param response_std:
    :param response_mutate_strength:
    :param response_mutate_rate:
    :param response_replace_rate:
    :param weight_mean:
    :param weight_std:
    :param weight_mutate_strength:
    :param weight_mutate_rate:
    :param weight_replace_rate:
    :param act_list:
    :param act_replace_rate:
    :param agg_list:
    :param agg_replace_rate:
    :param enabled_reverse_rate:
    :param add_node_rate:
    :param delete_node_rate:
    :param add_connection_rate:
    :param delete_connection_rate:
    :param single_structure_mutate: a genome is structurally mutate at most once
    :return:
    """

    # mutate_structure
    def nothing(rk, n, c):
        return n, c

    def m_add_node(rk, n, c):
        return mutate_add_node(rk, new_node_key, n, c, bias_default, response_default, act_default, agg_default)

    def m_delete_node(rk, n, c):
        return mutate_delete_node(rk, n, c, input_idx, output_idx)

    def m_add_connection(rk, n, c):
        return mutate_add_connection(rk, n, c, input_idx, output_idx)

    def m_delete_connection(rk, n, c):
        return mutate_delete_connection(rk, n, c)

    mutate_structure_li = [nothing, m_add_node, m_delete_node, m_add_connection, m_delete_connection]

    if single_structure_mutate:
        r1, r2, rand_key = jax.random.split(rand_key, 3)
        d = jnp.maximum(1, add_node_rate + delete_node_rate + add_connection_rate + delete_connection_rate)

        # shorten variable names for beauty
        anr, dnr = add_node_rate / d, delete_node_rate / d
        acr, dcr = add_connection_rate / d, delete_connection_rate / d

        r = rand(r1)
        branch = 0
        branch = jnp.where(r <= anr, 1, branch)
        branch = jnp.where((anr < r) & (r <= anr + dnr), 2, branch)
        branch = jnp.where((anr + dnr < r) & (r <= anr + dnr + acr), 3, branch)
        branch = jnp.where((anr + dnr + acr) < r & r <= (anr + dnr + acr + dcr), 4, branch)
        nodes, connections = jax.lax.switch(branch, mutate_structure_li, (r2, nodes, connections))
    else:
        r1, r2, r3, r4, rand_key = jax.random.split(rand_key, 5)

        # mutate add node
        aux_nodes, aux_connections = m_add_node(r1, nodes, connections)
        nodes = jnp.where(rand(r1) < add_node_rate, aux_nodes, nodes)
        connections = jnp.where(rand(r1) < add_node_rate, aux_connections, connections)

        # mutate delete node
        aux_nodes, aux_connections = m_delete_node(r2, nodes, connections)
        nodes = jnp.where(rand(r2) < delete_node_rate, aux_nodes, nodes)
        connections = jnp.where(rand(r2) < delete_node_rate, aux_connections, connections)

        # mutate add connection
        aux_nodes, aux_connections = m_add_connection(r3, nodes, connections)
        nodes = jnp.where(rand(r3) < add_connection_rate, aux_nodes, nodes)
        connections = jnp.where(rand(r3) < add_connection_rate, aux_connections, connections)

        # mutate delete connection
        aux_nodes, aux_connections = m_delete_connection(r4, nodes, connections)
        nodes = jnp.where(rand(r4) < delete_connection_rate, aux_nodes, nodes)
        connections = jnp.where(rand(r4) < delete_connection_rate, aux_connections, connections)

    nodes, connections = mutate_values(rand_key, nodes, connections, bias_mean, bias_std, bias_mutate_strength,
                                       bias_mutate_rate, bias_replace_rate, response_mean, response_std,
                                       response_mutate_strength, response_mutate_rate, response_replace_rate,
                                       weight_mean, weight_std, weight_mutate_strength,
                                       weight_mutate_rate, weight_replace_rate, act_list, act_replace_rate, agg_list,
                                       agg_replace_rate, enabled_reverse_rate)

    return nodes, connections


# @jit
def mutate_values(rand_key: Array,
                  nodes: Array,
                  connections: Array,
                  bias_mean: float = 0,
                  bias_std: float = 1,
                  bias_mutate_strength: float = 0.5,
                  bias_mutate_rate: float = 0.7,
                  bias_replace_rate: float = 0.1,
                  response_mean: float = 1.,
                  response_std: float = 0.,
                  response_mutate_strength: float = 0.,
                  response_mutate_rate: float = 0.,
                  response_replace_rate: float = 0.,
                  weight_mean: float = 0.,
                  weight_std: float = 1.,
                  weight_mutate_strength: float = 0.5,
                  weight_mutate_rate: float = 0.7,
                  weight_replace_rate: float = 0.1,
                  act_list: Array = None,
                  act_replace_rate: float = 0.1,
                  agg_list: Array = None,
                  agg_replace_rate: float = 0.1,
                  enabled_reverse_rate: float = 0.1) -> Tuple[Array, Array]:
    """
    Mutate values of nodes and connections.

    Args:
        rand_key: A random key for generating random values.
        nodes: A 2D array representing nodes.
        connections: A 3D array representing connections.
        bias_mean: Mean of the bias values.
        bias_std: Standard deviation of the bias values.
        bias_mutate_strength: Strength of the bias mutation.
        bias_mutate_rate: Rate of the bias mutation.
        bias_replace_rate: Rate of the bias replacement.
        response_mean: Mean of the response values.
        response_std: Standard deviation of the response values.
        response_mutate_strength: Strength of the response mutation.
        response_mutate_rate: Rate of the response mutation.
        response_replace_rate: Rate of the response replacement.
        weight_mean: Mean of the weight values.
        weight_std: Standard deviation of the weight values.
        weight_mutate_strength: Strength of the weight mutation.
        weight_mutate_rate: Rate of the weight mutation.
        weight_replace_rate: Rate of the weight replacement.
        act_list: List of the activation function values.
        act_replace_rate: Rate of the activation function replacement.
        agg_list: List of the aggregation function values.
        agg_replace_rate: Rate of the aggregation function replacement.
        enabled_reverse_rate: Rate of reversing enabled state of connections.

    Returns:
        A tuple containing mutated nodes and connections.
    """

    k1, k2, k3, k4, k5, rand_key = jax.random.split(rand_key, num=6)
    bias_new = mutate_float_values(k1, nodes[:, 1], bias_mean, bias_std,
                                   bias_mutate_strength, bias_mutate_rate, bias_replace_rate)
    response_new = mutate_float_values(k2, nodes[:, 2], response_mean, response_std,
                                       response_mutate_strength, response_mutate_rate, response_replace_rate)
    weight_new = mutate_float_values(k3, connections[0, :, :], weight_mean, weight_std,
                                     weight_mutate_strength, weight_mutate_rate, weight_replace_rate)
    act_new = mutate_int_values(k4, nodes[:, 3], act_list, act_replace_rate)
    agg_new = mutate_int_values(k5, nodes[:, 4], agg_list, agg_replace_rate)

    # refactor enabled
    r = jax.random.uniform(rand_key, connections[1, :, :].shape)
    enabled_new = connections[1, :, :] == 1
    enabled_new = jnp.where(r < enabled_reverse_rate, ~enabled_new, enabled_new)
    enabled_new = jnp.where(~jnp.isnan(connections[0, :, :]), enabled_new, jnp.nan)

    nodes = nodes.at[:, 1].set(bias_new)
    nodes = nodes.at[:, 2].set(response_new)
    nodes = nodes.at[:, 3].set(act_new)
    nodes = nodes.at[:, 4].set(agg_new)
    connections = connections.at[0, :, :].set(weight_new)
    connections = connections.at[1, :, :].set(enabled_new)
    return nodes, connections


# @jit
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


# @jit
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


# @jit
def mutate_add_node(rand_key: Array, new_node_key: int, nodes: Array, connections: Array,
                    default_bias: float = 0, default_response: float = 1,
                    default_act: int = 0, default_agg: int = 0) -> Tuple[Array, Array]:
    """
    Randomly add a new node from splitting a connection.
    :param rand_key:
    :param new_node_key:
    :param nodes:
    :param connections:
    :param default_bias:
    :param default_response:
    :param default_act:
    :param default_agg:
    :return:
    """
    # randomly choose a connection
    from_key, to_key, from_idx, to_idx = choice_connection_key(rand_key, nodes, connections)

    def nothing():
        return nodes, connections

    def successful_add_node():
        # disable the connection
        new_nodes, new_connections = nodes, connections
        new_connections = new_connections.at[1, from_idx, to_idx].set(False)

        # add a new node
        new_nodes, new_connections = \
            add_node(new_node_key, new_nodes, new_connections,
                     bias=default_bias, response=default_response, act=default_act, agg=default_agg)
        new_idx = fetch_first(new_nodes[:, 0] == new_node_key)

        # add two new connections
        weight = new_connections[0, from_idx, to_idx]
        new_nodes, new_connections = add_connection_by_idx(from_idx, new_idx,
                                                           new_nodes, new_connections, weight=1., enabled=True)
        new_nodes, new_connections = add_connection_by_idx(new_idx, to_idx,
                                                           new_nodes, new_connections, weight=weight, enabled=True)
        return new_nodes, new_connections

    # if from_idx == I_INT, that means no connection exist, do nothing
    nodes, connections = jax.lax.cond(from_idx == I_INT, nothing, successful_add_node)

    return nodes, connections


# @jit
def mutate_delete_node(rand_key: Array, nodes: Array, connections: Array,
                       input_keys: Array, output_keys: Array) -> Tuple[Array, Array]:
    """
    Randomly delete a node. Input and output nodes are not allowed to be deleted.
    :param rand_key:
    :param nodes:
    :param connections:
    :param input_keys:
    :param output_keys:
    :return:
    """
    # randomly choose a node
    node_key, node_idx = choice_node_key(rand_key, nodes, input_keys, output_keys,
                                         allow_input_keys=False, allow_output_keys=False)

    def nothing():
        return nodes, connections

    def successful_delete_node():
        # delete the node
        aux_nodes, aux_connections = delete_node_by_idx(node_idx, nodes, connections)

        # delete connections
        aux_connections = aux_connections.at[:, node_idx, :].set(jnp.nan)
        aux_connections = aux_connections.at[:, :, node_idx].set(jnp.nan)

        return aux_nodes, aux_connections

    nodes, connections = jax.lax.cond(node_idx == I_INT, nothing, successful_delete_node)

    return nodes, connections


# @jit
def mutate_add_connection(rand_key: Array, nodes: Array, connections: Array,
                          input_keys: Array, output_keys: Array) -> Tuple[Array, Array]:
    """
    Randomly add a new connection. The output node is not allowed to be an input node. If in feedforward networks,
    cycles are not allowed.
    :param rand_key:
    :param nodes:
    :param connections:
    :param input_keys:
    :param output_keys:
    :return:
    """
    # randomly choose two nodes
    k1, k2 = jax.random.split(rand_key, num=2)
    from_key, from_idx = choice_node_key(k1, nodes, input_keys, output_keys,
                                         allow_input_keys=True, allow_output_keys=True)
    to_key, to_idx = choice_node_key(k2, nodes, input_keys, output_keys,
                                     allow_input_keys=False, allow_output_keys=True)

    def successful():
        new_nodes, new_connections = add_connection_by_idx(from_idx, to_idx, nodes, connections)
        return new_nodes, new_connections

    def already_exist():
        new_connections = connections.at[1, from_idx, to_idx].set(True)
        return nodes, new_connections

    def cycle():
        return nodes, connections

    is_already_exist = ~jnp.isnan(connections[0, from_idx, to_idx])
    is_cycle = check_cycles(nodes, connections, from_idx, to_idx)

    choice = jnp.where(is_already_exist, 0, jnp.where(is_cycle, 1, 2))
    nodes, connections = jax.lax.switch(choice, [already_exist, cycle, successful])
    return nodes, connections


# @jit
def mutate_delete_connection(rand_key: Array, nodes: Array, connections: Array):
    """
    Randomly delete a connection.
    :param rand_key:
    :param nodes:
    :param connections:
    :return:
    """
    # randomly choose a connection
    from_key, to_key, from_idx, to_idx = choice_connection_key(rand_key, nodes, connections)

    def nothing():
        return nodes, connections

    def successfully_delete_connection():
        return delete_connection_by_idx(from_idx, to_idx, nodes, connections)

    nodes, connections = jax.lax.cond(from_idx == I_INT, nothing, successfully_delete_connection)

    return nodes, connections


# @partial(jit, static_argnames=('allow_input_keys', 'allow_output_keys'))
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


# @jit
def choice_connection_key(rand_key: Array, nodes: Array, connection: Array) -> Tuple[Array, Array, Array, Array]:
    """
    Randomly choose a connection key from the given connections.
    :param rand_key:
    :param nodes:
    :param connection:
    :return: from_key, to_key, from_idx, to_idx
    """

    k1, k2 = jax.random.split(rand_key, num=2)

    has_connections_row = jnp.any(~jnp.isnan(connection[0, :, :]), axis=1)

    def nothing():
        return jnp.nan, jnp.nan, I_INT, I_INT

    def has_connection():
        f_idx = fetch_random(k1, has_connections_row)
        col = connection[0, f_idx, :]
        t_idx = fetch_random(k2, ~jnp.isnan(col))
        f_key, t_key = nodes[f_idx, 0], nodes[t_idx, 0]
        return f_key, t_key, f_idx, t_idx

    from_key, to_key, from_idx, to_idx = jax.lax.cond(jnp.any(has_connections_row), has_connection, nothing)
    return from_key, to_key, from_idx, to_idx


# @jit
def rand(rand_key):
    return jax.random.uniform(rand_key, ())
