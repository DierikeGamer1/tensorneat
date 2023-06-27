import jax
from jax import Array, numpy as jnp
from jax import jit, vmap

from .utils import I_INT


def create_forward(config):
    """
    meta method to create forward function
    """

    def act(idx, z):
        """
        calculate activation function for each node
        """
        idx = jnp.asarray(idx, dtype=jnp.int32)
        # change idx from float to int
        res = jax.lax.switch(idx, config['activation_funcs'], z)
        return res

    def agg(idx, z):
        """
        calculate activation function for inputs of node
        """
        idx = jnp.asarray(idx, dtype=jnp.int32)

        def all_nan():
            return 0.

        def not_all_nan():
            return jax.lax.switch(idx, config['aggregation_funcs'], z)

        return jax.lax.cond(jnp.all(jnp.isnan(z)), all_nan, not_all_nan)

    def forward(inputs: Array, cal_seqs: Array, nodes: Array, cons: Array) -> Array:
        """
        jax forward for single input shaped (input_num, )
        nodes, connections are a single genome

        :argument inputs: (input_num, )
        :argument cal_seqs: (N, )
        :argument nodes: (N, 5)
        :argument connections: (2, N, N)

        :return (output_num, )
        """

        input_idx = config['input_idx']
        output_idx = config['output_idx']

        N = nodes.shape[0]
        ini_vals = jnp.full((N,), jnp.nan)
        ini_vals = ini_vals.at[input_idx].set(inputs)

        weights = jnp.where(jnp.isnan(cons[1, :, :]), jnp.nan, cons[0, :, :])  # enabled

        def cond_fun(carry):
            values, idx = carry
            return (idx < N) & (cal_seqs[idx] != I_INT)

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def hit():
                ins = values * weights[:, i]
                z = agg(nodes[i, 4], ins)  # z = agg(ins)
                z = z * nodes[i, 2] + nodes[i, 1]  # z = z * response + bias
                z = act(nodes[i, 3], z)  # z = act(z)

                new_values = values.at[i].set(z)
                return new_values

            def miss():
                return values

            # the val of input nodes is obtained by the task, not by calculation
            values = jax.lax.cond(jnp.isin(i, input_idx), miss, hit)
            return values, idx + 1

        vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        return vals[output_idx]

    return forward
