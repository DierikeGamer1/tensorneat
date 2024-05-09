import jax.numpy as jnp
from brax import envs
import classeshort
from .rl_jit import RLEnv


class BraxEnv(RLEnv):
    def __init__(self, env_name: str = "ant", backend: str = "generalized"):
        super().__init__()
        self.env = envs.create(env_name=env_name, backend=backend)

    @property
    def input_shape(self):
        return 202

    @property
    def output_shape(self):
        return 3

    def show(self, randkey, state, act_func, params, *args, **kwargs):

        import jax
        import imageio
        import numpy as np
        from brax.io import image
        from tqdm import tqdm
       
        obs, env_state = self.reset(randkey)
        reward, Final = 0.0, False
        state_histories = []

        def step(key, env_state, obs):
            key, _ = jax.random.split(key)
            action = act_func(state, obs, params)
            reward, Final, state = self.ambient.actions(action, ClosePrice)
            state.extend(ClosePrice[indice - 200 : indice])
            obs, env_state, r, Final, _ = self.step(randkey, env_state, action)
            return key, env_state, obs, r, Final

        while not Final:
            state_histories.append(env_state.pipeline_state)
            key, env_state, obs, r, Final = jax.jit(step)(randkey, env_state, obs)
            reward += r
        
        print("Total reward: ", reward)


