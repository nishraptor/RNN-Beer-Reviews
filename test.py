import numpy as np
def char2oh(str):
    alphabet  = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-\",:$%!();.[]?+/'{}@ """
    for i in range(len(str)):
        if str[i] not in alphabet:
            s = list(str)
            s[i] = '@'
            str = "".join(s)

    vector = [[0 if char != letter else 1 for char in alphabet]
              for letter in str]
    return np.array(vector)

def oh2char(vector):
    alphabet  = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-\",:$%!();.[]?+/'{}@ """
    str = ''
    for i in vector:
        for j in range(len(i)):
            if i[j] != 0:
                str += alphabet[j]
    return str

print()