import random
import numpy as np
from math import exp


def setWeight(neurons: int, input: int):
    weight = []
    for i in range(neurons):
        buffer = []
        for j in range(input):
            buffer.append(random.uniform(-1, 1))
        weight.append(buffer)
    return np.array(weight)

def ELU(alpha: float, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < 0:
                matrix[i][j] = alpha * (exp(matrix[i][j]) - 1)
    return matrix

def xELU(alpha: float, element):
    if element < 0:
        element = alpha * (exp(element) - 1)
    return element

def dELU(alpha: float, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < 0:
                matrix[i][j] = xELU(alpha, matrix[i][j]) + alpha
            else:
                matrix[i][j] = 1
    return matrix


def setOffset(neurons: int):
    weight = []
    for i in range(neurons):
        lst = []
        for j in range(1):
            lst.append(random.uniform(-1, 1))
        weight.append(lst)
    return np.array(weight)


def countError(reference, output):
    error = abs(reference ** 2) + abs(output ** 2)
    error -= 2 * reference * output
    error /= 2
    return error


def secondSWeight(w2, ratio, output, oerror, hidden):
    er = output - oerror
    out = ratio * er
    out *= hidden
    out = w2 - out.T
    return out


def secondFWeight(w1, ratio, output, oerror, w2, f, input):
    er = output - oerror
    out = ratio * er
    out *= w2
    temp = f @ input.T
    out = out @ temp
    out = w1 - out
    return out


def secondTWeight(w3, ratio, output, oerror, w2, f):
    er = output - oerror
    out = ratio * er
    out *= w2
    temp = f * output
    out = out @ temp
    out = w3 - out
    return out


def secondOffset(offset, output, oerror):
    result = offset + (output - oerror)
    return result



def learning(alpha: float, numbers, ref_vector: list, table: list, sequence: list):
    context = np.array([[0], [0], [0], [0]])
    w1, w2, w3, offset = setWeight(len(ref_vector), len(table[0])), setWeight(numbers, 4), setWeight(4, 4), setOffset(4)
    iterations = 0
    while iterations < 500:
        for i in range(len(table)):
            iterations += 1
            input = np.array([table[i]])
            input = input.T
            context = w3 @ context
            hide_layer = w1 @ input
            hide_layer = hide_layer + context
            hide_layer = hide_layer - offset
            oELU = hide_layer
            hide_layer = ELU(alpha, hide_layer)
            context = hide_layer
            output = w2 @ hide_layer
            elman_error = abs(countError(ref_vector[i], context[0][0]))
            print("-" * 15 + f"Iteration - {iterations}" + "-" * 15)
            print(f"Error = {elman_error}")
            print(f"Sequence - {sequence}")
            print(f"Trainable sequence: {table[i]}")
            print(f"Reference is {ref_vector[i]}, elman output - {output[0][0]}")
            print("-" * 45)
            w1, w2, w3, offset = secondFWeight(w1, .000001, output[0][0], ref_vector[i], w2, dELU(alpha, oELU), input),\
            secondSWeight(w2, .00000001, output[0][0], ref_vector[i], hide_layer),\
            secondTWeight(w3, .00000001, output[0][0], ref_vector[i], w2, dELU(alpha, oELU)),\
            secondOffset(offset, output[0][0], ref_vector[i])
amount_out = 1
table1 = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
ref_vector1 = [5, 6, 7, 8]
s1 = [1, 2, 3, 4, 5, 6, 7, 8]
table2 = [[0, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 5], [2, 3, 5, 8]]
ref_vector2 = [3, 5, 8, 13]
s2 = [0, 1, 1, 2, 3, 5, 8, 13]
table3 = [[1, 2, 4, 8], [2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64]]
ref_vector3 = [16, 32, 64, 128]
s3 = [1, 2, 4, 8, 16, 32, 64, 128]
learning(.25, amount_out, ref_vector=ref_vector1, table=table1, sequence= s1)