import flax
from flax import linen as nn
from flax.linen.activation import silu
from typing import Sequence


class MLP(nn.Module):
    '''
    The object is used to define the nn structure. Here, only the number of neurons in the hidden and output layer are required and that of input layer will be infered from the input array.
    nl: list
        defines the nn structure (only the hidden layers). Examples: [32,32]

    out: int32/int64
        represents the number of neurons in output layer. Example: 1 
    '''
    nl: Sequence[int]    # The nl is the structure of the nn but do not include the input layer
    nout: int
    def setup(self):
        self.nn=[nn.Dense(neuron) for neuron in self.nl]
        self.output=nn.Dense(self.nout)

    def __call__(self,x):
        for layer in self.nn:
            x=silu(layer(x))
        return self.output(x)
