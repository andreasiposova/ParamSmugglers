import math
from io import BytesIO

import os
import shutil
import tempfile
import zlib
from base64 import b64encode

import Crypto
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

'''
def encrypt_data(plain_text, n_lsbs, limit):
    encoded_text = ''
    n = len(plain_text)  # length of compressed data
    for i in range(0, n, 16):
        start = i
        end = i + 16
        if end < n:
            cipher_text = ENC.encrypt(plain_text[start:end])
        else:
            pad = 16 - (n - start)
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
'''

def round_down_divisible_by_16_for_encryption(num):
    return (num // 16) * 16
def encrypt_data(plain_text, ENC):
    #THE LENGTH OF PLAIN TEXT TO ENCODE NEEDS TO BE DIVISIBLE BY 16 WHEN REPRESENTED AS BITS
    #THE RESULTING GZIP FILE THAT IS TO BE ENCRYPTED HERE THUS NEEDS TO BE:
    # - smaller than num_params*n_lsbs (in bits)
    # - divisible by 16 for the encyption/decryption (bytes)
    ENC = AES.new('1234567812345678'.encode("utf8") * 2, AES.MODE_CBC, 'This is an IV456'.encode("utf8"))
    #DEC = AES.new('1234567812345678'.encode("utf8") * 2, AES.MODE_CBC, 'This is an IV456'.encode("utf8"))
    encoded_text = b''
    n = len(plain_text)  # length of compressed data
    for i in range(0, n, 16):
        start = i
        end = i + 16
        if end < n:
            cipher_text = ENC.encrypt(plain_text[start:end])
        else:
            pad = 16 - (n - start)
            cipher_text = ENC.encrypt(plain_text[start:] + b'0' * pad)
        encoded_text += cipher_text

    data = np.frombuffer(encoded_text, dtype=np.uint8)
    data = np.unpackbits(data)
    bits_of_data = data.tolist()  # Convert the NumPy array to a list of integers
    binary_string = ''.join(str(bit) for bit in bits_of_data)  # Join the list of integers into a single string

    return binary_string

def decrypt_data(binary_string, ENC):
    # Convert the binary string back into a NumPy array of bits
    bits_of_data = np.array([int(bit) for bit in binary_string], dtype=np.uint8)

    # Pack the bits back into a NumPy array of bytes
    data = np.packbits(bits_of_data)
    encoded_text = data.tobytes()

    plain_text = b''
    n = len(encoded_text)  # length of encrypted data
    for i in range(0, n, 16):
        start = i
        end = i + 16
        if end <= n:
            decipher_text = ENC.decrypt(encoded_text[start:end])
        else:
            pad = 16 - (n - start)
            decipher_text = ENC.decrypt(encoded_text[start:end])[:-pad]
        plain_text += decipher_text

    return plain_text


def gzip_compress_tabular_data(raw_data, limit):
    # compress the raw data and encrypt it
    # return the bit values whose LSBs are the cipher-text
    # raw data bit string
    # limit in bit

    comp_buff = BytesIO()
    with gzip.GzipFile(fileobj=comp_buff, mode="w") as f:
        f.write(raw_data.encode())
    compressed_data = comp_buff.getvalue()

    return compressed_data

import gzip
from io import BytesIO

"""
def compress_binary_string(raw_data, limit, n_cols):
    # Convert the binary string to bytes
    raw_data_bytes = int(raw_data, 2).to_bytes((len(raw_data) + 7) // 8, 'big')

    # Write the bytes to a buffer
    buff = BytesIO(raw_data_bytes)

    # Compress the buffer using gzip
    comp_buff = BytesIO()
    with gzip.GzipFile(fileobj=comp_buff, mode="wb") as f:
        f.write(buff.getvalue())

    # Check the size of the compressed data
    compressed_data = comp_buff.getvalue()
    compressed_data_size_in_bits = len(compressed_data) * 8
    truncated_raw_data = raw_data
    n_rows_to_hide = len(truncated_raw_data) / (n_cols * 32)

    if compressed_data_size_in_bits > limit:
        # Calculate the approximate ratio of raw data size to compressed data size
        ratio = len(raw_data) / compressed_data_size_in_bits

        # Estimate how much raw data you need to keep to achieve the desired compressed data size
        estimated_raw_data_size = int(ratio * limit)

        # Truncate the raw data to the estimated size
        truncated_raw_data = raw_data[:estimated_raw_data_size]

        # Recompress the truncated raw data
        compressed_data = compress_binary_string(truncated_raw_data, limit, n_cols)

    return compressed_data, n_rows_to_hide
"""
def compress_binary_string(raw_data, limit, n_cols):
    def _recursive_compress(raw_data, limit, n_cols):
        # Convert the binary string to bytes
        raw_data_bytes = int(raw_data, 2).to_bytes((len(raw_data) + 7) // 8, 'big')
        # Write the bytes to a buffer
        buff = BytesIO(raw_data_bytes)
        # Compress the buffer using gzip
        comp_buff = BytesIO()
        with gzip.GzipFile(fileobj=comp_buff, mode="wb") as f:
            f.write(buff.getvalue())
        truncated_raw_data = raw_data
        n_rows_bits_cap = len(truncated_raw_data)
        n_rows_to_hide = len(truncated_raw_data) / (n_cols * 32)
        n_rows_to_hide = math.floor(n_rows_to_hide)
        required_len = (n_rows_to_hide*32*n_cols) #required length of raw data given the number of rows that fit the limit calculated recursively

        #required_len = ((round_down_divisible_by_16_for_encryption(required_len))*8)
        #truncated_raw_data = truncated_raw_data[:required_len]
        # Check the size of the compressed data
        new_limit = round_down_divisible_by_16_for_encryption(limit)
        compressed_data = comp_buff.getvalue()
        compressed_data_size_in_bits = len(compressed_data) * 8
        n_rows_bits_cap = compressed_data_size_in_bits
        #required_compressed_data_size_in_bytes = round_down_divisible_by_16_for_encryption(len(compressed_data))
        #required_compressed_data_size_in_bits = required_compressed_data_size_in_bytes*8

        if compressed_data_size_in_bits < new_limit:
            #if compressed_data_size_in_bits < required_compressed_data_size_in_bits:
            #truncated_raw_data = truncated_raw_data[:required_len]
            #truncated_raw_data_bytes = int(truncated_raw_data, 2).to_bytes((len(truncated_raw_data) + 7) // 8, 'big')
            #truncated_raw_data = truncated_raw_data[:compressed_data_size_in_bits]
            # Convert the binary string to bytes
            #raw_data_bytes = int(truncated_raw_data, 2).to_bytes((len(truncated_raw_data) + 7) // 8, 'big')
            # Write the bytes to a buffer
            #buff = BytesIO(truncated_raw_data_bytes)
            # Compress the buffer using gzip
            #comp_buff = BytesIO()
            #with gzip.GzipFile(fileobj=comp_buff, mode="wb") as f:
            #    f.write(buff.getvalue())
            #compressed_data = comp_buff.getvalue()
            #if compressed_data_size_in_bits > limit:
        #if compressed_data_size_in_bits > required_compressed_data_size_in_bits:
        #    ratio = len(raw_data) / required_compressed_data_size_in_bits
        #    estimated_raw_data_size = int(ratio * required_compressed_data_size_in_bits)
        #    # Truncate the raw data to the estimated size
        #    truncated_raw_data = raw_data[:estimated_raw_data_size]
        #    return _recursive_compress(truncated_raw_data, limit, n_cols)
            return compressed_data, n_rows_to_hide, n_rows_bits_cap
        if compressed_data_size_in_bits > new_limit:
            # Calculate the approximate ratio of raw data size to compressed data size
            ratio = len(raw_data) / compressed_data_size_in_bits

            # Estimate how much raw data you need to keep to achieve the desired compressed data size
            estimated_raw_data_size = int(ratio * new_limit)

            # Truncate the raw data to the estimated size
            truncated_raw_data = raw_data[:estimated_raw_data_size]
            #truncated_raw_data = truncated_raw_data[:required_len]

            # Recursively compress the truncated raw data
            return _recursive_compress(truncated_raw_data, limit, n_cols)
        else:
            return compressed_data, n_rows_to_hide, n_rows_bits_cap

    return _recursive_compress(raw_data, limit, n_cols)

def decompress_gzip(compressed_data):
    # Decompress the compressed_data using gzip
    comp_buff = BytesIO(compressed_data)
    with gzip.GzipFile(fileobj=comp_buff, mode="rb") as f:
        decompressed_data = f.read()

    # Write the decompressed data to a buffer
    buff = BytesIO(decompressed_data)

    # Convert the bytes back into a binary string
    decompressed_data_bytes = buff.getvalue()
    binary_data = ''.join(f'{byte:08b}' for byte in decompressed_data_bytes)

    return binary_data


def binary_to_bytearray(binary_string, limit):
    # Process the binary string in chunks
    chunk_size = 1024 * 8  # 1024 bytes (8096 bits) per chunk
    byte_repr = bytearray()

    for i in range(0, min(limit, len(binary_string)), chunk_size):
        chunk = binary_string[i:i + chunk_size]

        # Pad the chunk with zeros if necessary
        padding = 8 - len(chunk) % 8
        if padding != 8:
            chunk = chunk.ljust(len(chunk) + padding, '0')

        # Split the chunk into 8-bit groups and convert to integers
        byte_integers = [int(chunk[j:j + 8], 2) for j in range(0, len(chunk), 8)]

        # Convert integers to bytes and append to the bytearray
        byte_repr.extend(bytes(byte_integers))

    return byte_repr


def rs_compress_and_encode(raw_data, limit, n_cols):
    def _recursive_rs_compress_and_encode(raw_data, limit, n_cols):
        truncated_raw_data = raw_data
        n_rows_bits_cap = len(truncated_raw_data)
        n_rows_to_hide = len(truncated_raw_data) / (n_cols * 32)
        n_rows_to_hide = math.floor(n_rows_to_hide)
        required_len = n_rows_to_hide * 32 * n_cols

        #truncated_raw_data = truncated_raw_data[:required_len]
        rs = RSCodec(10)  # You can adjust the number of ECC bytes based on the expected error rate
        data_bytes = truncated_raw_data.encode('utf-8')  # Convert the string to bytes using utf-8 encoding
        compressed_data = zlib.compress(data_bytes)
        ecc_encoded_data = rs.encode(compressed_data)
        binary_string = ''.join(format(byte, '08b') for byte in ecc_encoded_data)
        compressed_data_size_in_bits = len(binary_string)
        if compressed_data_size_in_bits < limit:
            truncated_raw_data = truncated_raw_data[:required_len]
            rs = RSCodec(10)  # You can adjust the number of ECC bytes based on the expected error rate
            data_bytes = truncated_raw_data.encode('utf-8')  # Convert the string to bytes using utf-8 encoding
            compressed_data = zlib.compress(data_bytes)
            ecc_encoded_data = rs.encode(compressed_data)
            binary_string = ''.join(format(byte, '08b') for byte in ecc_encoded_data)

        if compressed_data_size_in_bits > limit:
            # Calculate the approximate ratio of raw data size to compressed data size
            ratio = len(raw_data) / compressed_data_size_in_bits

            # Estimate how much raw data you need to keep to achieve the desired compressed data size
            estimated_raw_data_size = int(ratio * limit)

            # Truncate the raw data to the estimated size
            truncated_raw_data = raw_data[:estimated_raw_data_size]
            if required_len < estimated_raw_data_size:
                truncated_raw_data = truncated_raw_data[:required_len]

            # Recursively compress the truncated raw data
            return _recursive_rs_compress_and_encode(truncated_raw_data, limit, n_cols)
        else:
            return binary_string, n_rows_to_hide, n_rows_bits_cap

    return _recursive_rs_compress_and_encode(raw_data, limit, n_cols)


def rs_decode_and_decompress(binary_string):
    # Convert the binary string to a bytearray
    ecc_encoded_data = bytearray(int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8))

    # Reed-Solomon decoding
    rs = RSCodec(10)
    compressed_data = rs.decode(ecc_encoded_data)[0] # Unpack the tuple to get the compressed_data

    # Decompress the bytearray
    decompressed_data = zlib.decompress(compressed_data)

    # Decode the decompressed bytes into a string
    raw_data = decompressed_data.decode('utf-8')

    return raw_data

# Example usage
#original_data = b"Your binary string here"
#compressed_and_encoded_data = rs_compress_and_encode(original_data)
# Introduce some bit errors manually or through a noisy channel
#data_with_bit_errors = compressed_and_encoded_data

#recovered_data = decode_and_decompress(data_with_bit_errors)

#if recovered_data == original_data:
#    print("Data successfully recovered")
#else:
#    print("Data recovery failed")

