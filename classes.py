from cv2 import cv2
import numpy as np
from scipy import special


def fun_active(x):
    return special.expit(x)


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


def query(input_hidden_w, hidden_out_w, inputs_list):
    inputs_sig = np.array(inputs_list, ndmin = 2).T
    hidden_inputs = np.dot(input_hidden_w, inputs_sig)
    hidden_out = fun_active(hidden_inputs)
    final_inputs = np.dot(hidden_out_w, hidden_out)
    final_out = fun_active(final_inputs)
    return final_out


def treyn(target_list, input_list, input_hidden_w, hidden_out_w, learn_node):
    targgets = np.array(target_list, ndmin=2).T
    inputs_sig = np.array(input_list, ndmin=2).T
    hidden_inputs = np.dot(input_hidden_w, inputs_sig)
    hidden_out = fun_active(hidden_inputs)
    final_inputs = np.dot(hidden_out_w, hidden_out)
    final_out = fun_active(final_inputs)
    out_errors = targgets - final_out
    hidden_errors = np.dot(hidden_out_w.T, out_errors)
    hidden_out_w += learn_node * np.dot((out_errors * final_out * (1 - final_out)), np.transpose(hidden_out))
    input_hidden_w += learn_node * np.dot((hidden_errors * hidden_out * (1 - hidden_out), np.transpose(inputs_sig)))


