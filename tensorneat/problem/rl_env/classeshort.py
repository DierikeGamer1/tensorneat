import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore
from jax import device_get,lax,jit # type: ignore
import jax   # type: ignore
from typing import Any, Dict, Optional, Tuple, Union
import sys
import chex # type: ignore
from flax import struct # type: ignore


class AmbientDeTreino:
    global contLong, contShort

    def __init__(self, ClosePrice, printar=False):
        self.position_short = False
        self.position_long = False
        self.Final = False
        self.reward = 0
        self.ClosePrice = ClosePrice
        self.printar = printar

    # Método para resetar o estado do ambiente
    def reset(self):
        self.position_short = False
        self.position_long = False
        self.Final = False
        self.reward = 0

        input_list = [
            float(self.position_short),
            float(self.position_long),
            *self.ClosePrice[:200],
        ]
        return input_list

    def actions(self, action, ClosePrice):
        reward = 0
        action = jnp.argmax(action)

        def AbrirShort():
            reward, self.Final = self.OpenPosition("Short", BuyPrice=ClosePrice[-1])
            # jax.debug.print('AbrirShort')
            return float(reward), self.Final

        def CloseShort():
            reward, self.Final = self.ClosePosition("Short", SellPrice=ClosePrice)
            # jax.debug.print('CloseShort')
            return float(reward), self.Final

        def FazerNada():
            reward = 0 
            # jax.debug.print('FazerNada')
            return float(reward),self.Final

        functions_list = [
            AbrirShort,CloseShort,FazerNada
        ]
        reward,self.Final=jax.lax.switch(action, functions_list)

        return (
            reward,
            self.Final,
            [float(self.position_long), float(self.position_short)],
        )

    def OpenPosition(self, direction, BuyPrice):
        reward = 0
        if direction == "Short":
            if self.position_long or self.position_short:
                self.Final = True
            else:
                self.position_short = True
                self.PriceOpenShort = BuyPrice
                reward -= 0.0005
        return reward, self.Final

    def ClosePosition(self, DirectionPosition, SellPrice):
        reward = 0
        def Passar(SellPrice):
            return 1
        def Testar(SellPrice):
            self.Final = True
            return 1

        if DirectionPosition == "Short":
            if self.position_short == False:
                self.Final = True

            jax.lax.cond(SellPrice[-1] < SellPrice[-2],Testar,Passar,SellPrice)
            if self.Final == False:
                if self.PriceOpenShort > SellPrice[-1]:
                    reward += self.PriceOpenShort - SellPrice[-1]
                else:
                    reward -= SellPrice[-1] - self.PriceOpenShort
            self.position_short = False
        return reward, self.Final

    def FecharAcabouDados(self, SellPrice, indice):
        reward = 0
        
        if self.position_short:
            if self.PriceOpenShort > SellPrice[indice]:
                reward += self.PriceOpenShort - SellPrice[indice]
            else:
                reward -= SellPrice[indice] - self.PriceOpenShort

        return reward
