from io import BytesIO

import os
import shutil
import tempfile
import zlib
from base64 import b64encode

from Crypto.Cipher import AES
from PIL import Image
from io import StringIO as BytesIO, StringIO
from io import BytesIO
import numpy as np
import gzip
import zlib
from reedsolo import RSCodec

ENC = AES.new('1234567812345678'.encode("utf8") * 2, AES.MODE_CBC, 'This is an IV456'.encode("utf8"))
DEC = AES.new('1234567812345678'.encode("utf8") * 2, AES.MODE_CBC, 'This is an IV456'.encode("utf8"))

def encrypt_data(plain_text, n_lsbs, limit):
    encoded_text = ''
    n = len(plain_text)  # length of compressed data
    for i in range(0, n, n_lsbs):
        start = i
        end = i + n_lsbs
        if end < n:
            cipher_text = ENC.encrypt(plain_text[start:end])
        else:
            pad = n_lsbs - (n - start)
            cipher_text = ENC.encrypt(plain_text[start:] + b'0' * pad)
        encoded_text += cipher_text.decode('latin-1')

    data = list(map(int, encoded_text))
    data = np.asarray(data, dtype=np.uint8)
    data = np.unpackbits(data)
    bits_of_data = data

    assert limit >= len(data)
    data = data.reshape(-1, n_lsbs)
    encrypted_data = np.asarray(data, dtype=np.uint8)
    return encrypted_data

def gzip_compress_tabular_data(raw_data):
    # compress the raw data and encrypt it
    # return the bit values whose LSBs are the cipher-text
    # raw data bit string
    # limit in bit

    comp_buff = BytesIO()
    with gzip.GzipFile(fileobj=comp_buff, mode="w") as f:
        f.write(raw_data.encode())
    compressed_data = comp_buff.getvalue()

    return compressed_data


def rs_compress_and_encode(data):
    rs = RSCodec(10)  # You can adjust the number of ECC bytes based on the expected error rate
    compressed_data = zlib.compress(data)
    ecc_encoded_data = rs.encode(compressed_data)
    return ecc_encoded_data

def decode_and_decompress(data_with_bit_errors):
    rs = RSCodec(10)
    corrected_data = rs.decode(data_with_bit_errors)
    decompressed_data = zlib.decompress(corrected_data)
    return decompressed_data

# Example usage
original_data = b"Your binary string here"
compressed_and_encoded_data = rs_compress_and_encode(original_data)

# Introduce some bit errors manually or through a noisy channel
data_with_bit_errors = compressed_and_encoded_data

recovered_data = decode_and_decompress(data_with_bit_errors)

if recovered_data == original_data:
    print("Data successfully recovered")
else:
    print("Data recovery failed")

