import copy
from random import uniform
from typing import List, Any, Union
from math import exp


class DimensionError(Exception):
    def __init__(self):
        super()


class TfNotDefinedError(Exception):
    pass


class Neurone:
    def __init__(self, n__x, b=0.0, tf='Heaviside'):
        self.__X = []
        # se n sono gli input, n+1 sono le lunghezze di X e W
        self.__n_X = n__x + 1
        self.__A = 0
        self.__n_Y = 1
        self.__W = []
        self.__b = b
        for i in range(self.__n_X - 1):
            self.__W.append(uniform(1.0, -1.0))
        self.__W.append(self.__b)
        #print('pesi iniziali: ', self.__W)
        self.__tf = tf
        self.__Y = 0.0
        self.__delta_w = 0.0

    def get_bias(self):
        return self.__b

    def get_weights(self, index):
        return self.__W[index]

    def get_delta(self):
        return self.__delta_w

    def correct_weights(self, error: float, neta: float = 0.6) -> float:
        #print('correzione pesi')
        self.__delta_w = error * self.__Y * (1 - self.__Y)
        #print('self.__X: ', self.__X)
        #print('self.__delta_w: ', self.__delta_w)
        W_old = copy.copy(self.__W)
        #print('W_old: ', W_old)
        self.__W = []
        for i in range(len(W_old)):
            self.__W.append(W_old[i] + neta * self.__delta_w * self.__X[i])
        #print('pesi: ', self.__W)

        self.__b = self.__W[-1]

    def set_input(self, X):
        #print('Calcolo output dato input')
        self.__X.clear()
        if not type(X) == list:
            raise TypeError()
        for e in X:
            if type(e) == complex:
                if e.imag == 0.0:
                    e = e.real
            if not (type(e) == int or type(e) == float):
                raise TypeError()
        if not self.__n_X == (len(X) + 1):
            raise DimensionError()
        # set degli input X + bias
        #print('X: ', X)
        self.__X.extend(X)
        self.__X.append(1)  # bias
        # computa y
        self.__compute_a()
        self.__compute_y()
        #print('self.__Y: ', self.__Y)
        return self.__Y

    def __compute_a(self):
        molt = []
        for i in range(len(self.__W)):
            molt.append(self.__W[i] * self.__X[i])
        # print('molt', molt)
        self.__A = 0.0
        for a in molt:
            self.__A = self.__A + a
        #print('self.__W: ', self.__W)
        #print('self.__A: ', self.__A)

    def __compute_y(self):
        if self.__tf == 'Heaviside':
            if self.__A >= 0:
                self.__Y = 1
            else:
                self.__Y = 0
        elif self.__tf == 'Sigmoidal':
            try:
                self.__Y = 1 / (1 + exp(-self.__A))
            except OverflowError:
                print("C'e stato un overflow")  # da cambiare
        else:
            raise TfNotDefinedError()
        # print('self.__Y: ', self.__Y)


class InputNotZeroError(Exception):
    pass


class LayerNeuralNet:
    __n_layers: List[int]

    # funziona
    def __init__(self, config):
        self.__global_config = config
        #print(self.__global_config)
        if not type(self.__global_config) == list:
            raise TypeError()
        else:
            for e in self.__global_config:
                if not type(e) == int:
                    raise TypeError()
        self.__n_input = self.__global_config[0]
        if self.__n_input == 0:
            raise InputNotZeroError()
        # print(self.__n_input)
        self.__X = None
        self.__n_output = self.__global_config[len(self.__global_config) - 1]
        # print(self.__n_output)
        # print('output ', self.__n_output)
        self.__n_hidden = self.__global_config[1:len(self.__global_config) - 1]
        # print('hidden ', self.__n_hidden)
        self.__n_layers = []
        self.__n_layers.extend(self.__n_hidden)
        self.__n_layers.append(self.__n_output)
        # print('layers', self.__n_layers)
        self.__Y = []
        layer = list()
        matrix = list()
        for index1 in range(len(self.__n_layers)):
            # print('index1 ', index1)
            n = self.__n_layers[index1]
            # print('index layer {} , n neuroni {}'.format(index1, n))
            # print('layer prima: ', layer)
            for i in range(n):
                layer.append(Neurone(self.__global_config[index1], b=uniform(-2.0, 2.0), tf='Sigmoidal'))
                # print('n ingressi neuroni: ', self.__global_config[index1], ', con ', n, ' neuroni')
            # print('layer dopo: ', layer)
            # print('fase ', index1,' matrice ', matrix)
            matrix.append(layer)
            # print('fase ', index1,' matrice ', matrix)
            layer = []
        # print('complete matrix: ', matrix)
        self.__perceptron_net = matrix
        # print(self.__perceptron_net)
        self.__output_layer = self.__perceptron_net[len(self.__perceptron_net) - 1]
        self.__hidden_layer = self.__perceptron_net[0:len(self.__perceptron_net) - 1]

    def set_net_input(self, X):
        # funziona
        input_layer = list()
        output_layer = list()
        input_layer.append(X)
        for layer in self.__perceptron_net:
            index = self.__perceptron_net.index(layer)
            perceptron: Neurone
            for perceptron in layer:
                out = perceptron.set_input(input_layer[index])
                output_layer.append(out)
            input_layer.append(output_layer)
            output_layer = []
        self.__X = input_layer[0:len(self.__n_layers)]  # ingressi dei layer
        self.__Y = input_layer[len(input_layer) - 1]  # ultimo elemento, uscita della rete
        for y in self.__Y:
            if type(y) == complex:
                if y.imag == 0.0:
                    pass
        return self.__Y

    def correct_net_weights(self, D: List[float], neta: float = 0.6):
        # creo gli errori
        err_y = []
        for i in range(len(self.__Y)):
            err_y.append(D[i] - self.__Y[i])
        err_layers = []
        # print('err_y: ', err_y)
        hidden_layer_rev = copy.copy(self.__hidden_layer)
        hidden_layer_rev.reverse()
        f = True
        for hidd_layer in hidden_layer_rev:
            index_layer = hidden_layer_rev.index(hidd_layer)
            # print('index layer: ', index_layer)
            # print('hidden layer[', index_layer, ']: ', hidd_layer)
            err_layer = []
            if f:
                target = self.__output_layer
                f = False
            else:
                target = hidden_layer_rev[index_layer - 1]
            # print('target: ', target)
            for neuron in hidd_layer:
                index = hidd_layer.index(neuron)
                error = 0.0
                for neu in target:
                    error = error + neu.get_delta() * neu.get_weights(index)
                err_layer.append(error)
            err_layers.append(err_layer)
        # correggo i pesi

        for neuron in self.__output_layer:
            index = self.__output_layer.index(neuron)
            neuron.set_input(self.__X[-1])
            neuron.correct_weights(err_y[index], neta)
        # i = 0
        f = True
        # hidden_layer_rev = copy.copy(self.__hidden_layer)
        # print('self.__hidden_layer: ', self.__hidden_layer)
        # hidden_layer_rev.reverse()
        # print('self.__hidden_layer: ', self.__hidden_layer)
        # print('hidden_layer_rev: ', hidden_layer_rev)
        for hidd_layer in hidden_layer_rev:
            index_layer = hidden_layer_rev.index(hidd_layer)
            # print('index layer: ', index_layer)
            # print('hidden layer[', index_layer, ']: ', hidd_layer)
            if f:
                target = self.__output_layer
                f = False
            else:
                target = hidden_layer_rev[index_layer - 1]

            self.__X.reverse()
            for neuron in hidd_layer:
                index = hidd_layer.index(neuron)
                # print(len(self.__hidden_layer))
                # print(len(self.__X))
                neuron.set_input(self.__X[index_layer + 1])
                neuron.correct_weights(err_layers[index_layer][index], neta)
            self.__X.reverse()


class Row:
    def __init__(self, input, output):
        self.input = input
        self.output = output


class DesiredTable:

    def __init__(self, row: Row = None):
        self.table = []
        if row is not None:
            self.table.append(row)

    def append(self, row):
        self.table.append(row)

    def clear(self):
        self.table = []


class Addestramento:

    def __init__(self, net, tab_D, neta, eps):
        self.__net = net
        self.__tabella_D = tab_D
        self.__neta = neta
        self.__eps = eps

    def addestra(self):
        n = -1
        y_rows = []
        D = []
        y = []
        while True:
            n = n + 1
            print(n)
            # clacolo uscite della rete
            for row in self.__tabella_D.table:
                #print('row.input: ', row.input, '; row.output: ', row.output)
                y = self.__net.set_net_input(row.input)
                y_rows.append(y)
                D.append(row.output)

            # controllo
            if self.__check_less_eps(D, y_rows, self.__eps):
                break
            else:
                for row in self.__tabella_D.table:
                    i = self.__tabella_D.table.index(row)
                    d = D[i]
                    self.__net.set_net_input(row.input)
                    self.__net.correct_net_weights(d, self.__neta)
            y_rows.clear()
            D.clear()

        return self.__net, n

    def __check_less_eps(self, ref, y, epsilon: float):
        """

        :type epsilon: float
        """
        cond = True
        error = 0.0
        error_cond = False
        for i in range(len(y)):
            for j in range(len(y[i])):
                error = (abs(ref[i][j] - y[i][j]))
                error_cond = error < epsilon
                print('error: ', error, '=', ref[i][j], ' - ', y[i][j])
                print('cond: ', error_cond)
                cond = cond and error_cond
        return cond


def test():
    # main
    configuration_net = [3, 1]
    print('configurations ', configuration_net)
    rete = LayerNeuralNet(configuration_net)
    # tabella di addestramento
    tabella_addestramento = DesiredTable()
    # print('input output')

    tabella_addestramento.append(Row([0, 0, 0], [0.1]))
    #tabella_addestramento.append(Row([1, 0, 0], [0.1]))
    tabella_addestramento.append(Row([0, 1, 0], [0.2]))
    tabella_addestramento.append(Row([0, 0, 1], [0.2]))
    #tabella_addestramento.append(Row([1, 1, 0], [0.8]))
    tabella_addestramento.append(Row([0, 1, 1], [0.8]))
    tabella_addestramento.append(Row([1, 0, 1], [0.8]))
    tabella_addestramento.append(Row([1, 1, 1], [0.9]))
    neta = 0.8
    eps = 0.1
    addestramento = Addestramento(rete, tabella_addestramento, neta, eps)
    rete_addestrata, n_addr = addestramento.addestra()
    X = [1, 1, 0]
    print('input: ', X)
    print('uscita finale: ', rete_addestrata.set_net_input(X), '; numero tentativi: ', n_addr)


if __name__ == '__main__':
    test()