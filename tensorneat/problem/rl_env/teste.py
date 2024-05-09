from functools import partial
import jax.numpy as jnp # type: ignore
import jax # type: ignore
import sys

import numpy as np # type: ignore
def falsef(c):
    return 2 
def truef(c):
    return 3,2 
c=3

c,g=jax.lax.cond(c>3,truef,lambda g: (2,3),c)
print(c,g)