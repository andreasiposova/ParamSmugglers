import struct
import numpy as np
from codecs import decode
import torch
def reconstruct_from_lsbs(bits_params, bits):
    data = np.asarray(bits_params, dtype=int)  # .reshape(total_params*8, 8)
    data = np.packbits(data.astype(np.uint8))
    data = list(map(chr, data))

    return data

def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]
def float2bin64(f):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!d', f))
def float2bin32(f):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f))

def bin2float64(binary):
    d = ''.join(str(x) for x in binary)
    bf = int_to_bytes(int(d, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def bin2float32(binary):
    #d = ''.join(str(x) for x in binary)
    bf = int_to_bytes(int(binary, 2), 4)  # 4 bytes needed for IEEE 754 binary32.
    return struct.unpack('>f', bf)[0]


def params_to_bits(params):
    params_as_bits = []
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            flattened_value = value.flatten()  # flatten the tensor
            for v in flattened_value:
                params_as_bits.extend(float2bin32(v))
    params_as_bits = ''.join(params_as_bits)
    return params_as_bits


"""def bits_to_params(params_as_bits, shape_dict):
    params = {}
    i = 0
    for key, shape in shape_dict.items():
        size = torch.prod(torch.tensor(shape)).item()  # calculate size of tensor
        bits = params_as_bits[i:i+size]         # extract bits for tensor
        i += size
        #tensor = torch.tensor([bin2float32(b) for b in bits]).reshape(shape)  # convert bits to tensor
        tensor = torch.tensor([bin2float32(bits) for b in bits]).reshape(shape)  # convert bits to tensor
        params[key] = tensor
    return params"""

def bits_to_params(params_as_bits, shape_dict):
    params = {}
    i = 0
    for key, shape in shape_dict.items():
        if 'num_batches_tracked' in key:
            size = 1
        else:# calculate size of tensor
            size = torch.prod(torch.tensor(shape)).item()  # calculate size of tensor
        bits = params_as_bits[i:i+size*32]
        len_bits = len(bits)# extract 32 bits at a time for tensor
        i += size*32
        tensor = torch.zeros(shape)
        for j in range(size):
            start = j*32
            end = (j+1)*32
            float_val = bin2float32(bits[start:end])
            tensor.view(-1)[j] = float_val
        params[key] = tensor
    return params