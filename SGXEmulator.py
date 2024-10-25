import logging
import time

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import pickle
import os


def pad_data(data, block_size):
    padder = padding.PKCS7(block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    return padded_data


def unpad_data(padded_data, block_size):
    unpadder = padding.PKCS7(block_size).unpadder()
    data = unpadder.update(padded_data)
    try:
        data += unpadder.finalize()
    except ValueError:
        # If the padding is incorrect, return the data as is
        pass
    return data


def encrypt_aes(key, plaintext):
    # Generate a random IV (Initialization Vector)
    iv = os.urandom(16)

    # Pad the plaintext to ensure its length is a multiple of the block size
    padded_plaintext = pad_data(plaintext, algorithms.AES.block_size)

    # Create AES cipher with CBC mode
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Encrypt the padded plaintext
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()

    # Return IV and ciphertext
    return iv, ciphertext


def decrypt_aes(key, iv, ciphertext):
    # Create AES cipher with CBC mode
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Decrypt the ciphertext
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the decrypted plaintext
    plaintext = unpad_data(padded_plaintext, algorithms.AES.block_size)

    # Return plaintext
    return plaintext


def store_model(key, transformer, filename='encrypted_model.bin'):
    # Create AES cipher with CBC mode
    serialized_model = pickle.dumps({'transformer':transformer})
    iv, encrypted_model = encrypt_aes(key, serialized_model)
    # Write the encrypted model to a file
    with open(filename, 'wb') as f:
        f.write(iv)  # Write the IV first
        f.write(encrypted_model)  # Write the encrypted model
    # Return plaintext
    return filename
def store_plain_model(transformer, filename='encrypted_model.bin'):
    serialized_model = pickle.dumps({'transformer':transformer})
    # Write the encrypted model to a file
    with open(filename, 'wb') as f:
        f.write(serialized_model)  # Write the encrypted model
    return filename


def load_model(file_path, key):
    with open(file_path, 'rb') as f:
        iv = f.read(16)  # Read the IV
        ciphertext = f.read()  # Read the encrypted model
    # Decrypt the model
    decrypted_model = decrypt_aes(key, iv, ciphertext)
    # Deserialize the decrypted model
    transformer_model = pickle.loads(decrypted_model)
    model = transformer_model['transformer']
    return model

# def load_plain_model(file_path):
#     with open(file_path, 'rb') as f:
#         data = f.read()  # Read the encrypted model
#     # Deserialize the decrypted model
#     transformer_model = pickle.loads(data)
#     model = transformer_model['transformer']
#     return model
# def load_plain_model(file_path):
#     print(f"file_path: {file_path}, type: {type(file_path)}")  # Debugging print statement
#     with open(file_path, 'rb') as f:
#         data = f.read()  # Read the encrypted model
#     # Deserialize the decrypted model
#     transformer_model = pickle.loads(data)
#     model = transformer_model['transformer']
#     return model

def load_plain_model(file_path):
        print(f"file_path: {file_path}, type: {type(file_path)}")  # Debugging print statement

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Check if the file is readable
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"The file {file_path} is not readable.")

        try:
            with open(file_path, 'rb') as f:
                data = f.read()  # Read the encrypted model
            # Deserialize the decrypted model
            transformer_model = pickle.loads(data)
            model = transformer_model['transformer']
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load the model from {file_path}: {e}")