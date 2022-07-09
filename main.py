from cv2 import cv2
import numpy as np


def init_net():
    input_nodes = 784
    print('Введите число скрытых нейронов')
    hidden_nodes = int(input())
    out_nodes = 10
    print('Введите скорость обучения')
    learn_node = float(input())
    return input_nodes, hidden_nodes, out_nodes, learn_node


def create_net(input_nodes, hidden_nodes, out_nodes):
    input_hidden_w = (np.random.rand(hidden_nodes, input_nodes) - 0.5)
    hidden_out_w = (np.random.rand(out_nodes, hidden_nodes) - 0.5)
    return input_hidden_w, hidden_out_w


#2


'''
Я супер крутой remkad зовут
'''

