import torch
import math
from torch import nn


# https://github.com/louislva/deepmind-perceiver/blob/eb668c818eceab957713091ad1e244b14f039f7e/perceiver/positional_encoding.py#L6
# Example parameters: shape=(28, 28), bands=8
# encoding_size = bands*2*2
def fourier_features(shape, bands):
    # This first "shape" refers to the shape of the input data, not the output of this function
    dims = len(shape)

    # Every tensor we make has shape: (bands, dimension, x, y, etc...)

    # Pos is computed for the second tensor dimension
    # (aptly named "dimension"), with respect to all
    # following tensor-dimensions ("x", "y", "z", etc.)
    pos = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
    )))
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

    # Band frequencies are computed for the first
    # tensor-dimension (aptly named "bands") with
    # respect to the index in that dimension
    band_frequencies = (torch.logspace(
        math.log(1.0),
        math.log(shape[0]/2),
        steps=bands,
        base=math.e
    )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)

    # For every single value in the tensor, let's compute:
    #             freq[band] * pi * pos[d]

    # We can easily do that because every tensor is the
    # same shape, and repeated in the dimensions where
    # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
    result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

    # Use both sin & cos for each band, and then add raw position as well
    # TODO: raw position
    result = torch.cat([
        torch.sin(result),
        torch.cos(result),
    ], dim=0)

    return result


    
    
    
