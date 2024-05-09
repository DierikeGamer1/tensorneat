from functools import partial
import jax.numpy as jnp # type: ignore
import jax # type: ignore
import sys
from .. import BaseProblem
import problem.rl_env.classeshort as classeshort
import numpy as np # type: ignore


class RLEnv(BaseProblem):
    jitable = True

    # TODO: move output transform to algorithm
    def __init__(self):
        super().__init__()
        

    def VerificarFinals(Final, indice, ListaClosePrice, env, SellPrice):
        reward = 0  
        def FecharCasoFinal(c=None):
            global Final
            Final = True
            return env.FecharAcabouDados(SellPrice, indice)
        def IrParaIndiceAcabou(c=None):
            return jax.lax.cond(indice == ListaClosePrice - 1 ,FecharIndiceAcabou,lambda: 0)
            
        def FecharIndiceAcabou(c=None):
            global Final
            Final = True
            return env.FecharAcabouDados(SellPrice, indice)
        reward = jax.lax.cond(Final==True,FecharCasoFinal,IrParaIndiceAcabou)
        
    
        return Final, reward

    def evaluate(self, randkey, state, act_func, params, ListaClosePrice):
        
        def cond_func(carry):
            _, done, _ = carry
            return ~done

        def body_func(carry):
            env_state, _, tr = carry  # total reward

            action = act_func(env_state, params) 
            reward, Final, next_env_state = env.actions(action, ClosePrice)
            # Convertendo os índices para inteiros concretos usando `int`
            start_index = self.indice - 200
            end_index = self.indice

            # Usando dynamic_slice para fatiar o array ClosePrice
            sliced_close_price = jax.lax.dynamic_slice(ClosePrice, (start_index, ), (end_index, ))

            next_env_state.extend(sliced_close_price)
            reward1 = 0
            Final, reward1 = RLEnv.VerificarFinals( 
                Final,
                self.indice,
                lenListaClosePrice,
                env,
                ClosePrice,           
            )
            reward += reward1
            self.indice+=1
            return next_env_state, Final, tr + reward

        for arquivo in ListaClosePrice:
            ClosePrice=jnp.asarray(arquivo)
            
            env = classeshort.AmbientDeTreino(arquivo)
            init_env_state = env.reset()
            self.indice = 201
            lenListaClosePrice = len(ListaClosePrice)

            _, _, total_rewardep = jax.lax.while_loop(
                cond_func, body_func, (init_env_state, False, 0.0)
            )
            total_reward+=total_rewardep
            jax.debug.print("Fim Arquivo")

        return total_reward

    @property
    def input_shape(self):
        return [202]

    @property
    def output_shape(self):
        return [3]

    def show(self, randkey, state, act_func, params, *args, **kwargs):
        raise NotImplementedError
