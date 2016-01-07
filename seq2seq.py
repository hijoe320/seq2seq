# -*- coding: utf-8 -*-

__author__ = "Haizhou Qu"

import numpy as np
import theano
import theano.tensor as T
from six.moves import zip

epsilon = 1e-6
dtype = theano.config.floatX


def shared(value, name=None):
    return theano.shared(value.astype(dtype), name=name)


def shared_zeros(shape, name=None):
    return shared(value=np.zeros(shape), name=name)


def shared_zeros_like(x, name=None):
    return shared_zeros(shape=x.shape, name=name)


def init_weights(shape, name=None):
    bound = np.sqrt(1.0/shape[1])
    w = np.random.uniform(-bound, bound, shape)
    return shared(value=w, name=name)


def adadelta(params, cost, lr=1.0, rho=0.95):
    # from https://github.com/fchollet/keras/blob/master/keras/optimizers.py
    grads = T.grad(cost, params)
    accus = [shared_zeros_like(p.get_value()) for p in params]
    delta_accus = [shared_zeros_like(p.get_value()) for p in params]
    updates = []
    for p, g, a, d_a in zip(params, grads, accus, delta_accus):
        new_a = rho * a + (1.0 - rho) * T.square(g)
        updates.append((a, new_a))
        update = g * T.sqrt(d_a + epsilon) / T.sqrt(new_a + epsilon)
        new_p = p - lr * update
        updates.append((p, new_p))
        new_d_a = rho * d_a + (1.0 - rho) * T.square(update)
        updates.append((d_a, new_d_a))
    return updates


def categorical_crossentropy(y_true, y_pred):
    # from https://github.com/fchollet/keras/blob/master/keras/objectives.py
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)

    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return T.mean(cce)


def mean_square_error(y_true, y_pred):
    return T.mean(T.square(y_pred - y_true))


class LSTM(object):
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim

        shape_b = (1, size)
        shape_U = (dim, size)
        shape_W = (size, size)

        self.h_tm1 = shared_zeros(shape_b, "h_tm1")
        self.c_tm1 = shared_zeros(shape_b, "c_tm1")

        self.Ui = init_weights(shape_U, "Ui")
        self.Wi = init_weights(shape_W, "Wi")
        self.bi = shared_zeros(shape_b, "bi")

        self.Uf = init_weights(shape_U, "Uf")
        self.Wf = init_weights(shape_W, "Wf")
        self.bf = shared_zeros(shape_b, "bf")

        self.Uo = init_weights(shape_U, "Uo")
        self.Wo = init_weights(shape_W, "Wo")
        self.bo = shared_zeros(shape_b, "bo")

        self.Ug = init_weights(shape_U, "Ug")
        self.Wg = init_weights(shape_W, "Wg")
        self.bg = shared_zeros(shape_b, "bg")

        self.params = [
            self.Ui, self.Wi, self.bi,
            self.Uf, self.Wf, self.bf,
            self.Uo, self.Wo, self.bo,
            self.Ug, self.Wg, self.bg
        ]

    def set_state(self, h, c):
        self.h_tm1.set_value(h.get_value())
        self.c_tm1.set_value(c.get_value())

    def reset_state(self):
        self.h_tm1 = shared_zeros((1, self.size), "h_tm1")
        self.c_tm1 = shared_zeros((1, self.size), "c_tm1")

    @staticmethod
    def step(
        x_t, h_tm1, c_tm1,
        Ui, Wi, bi, Uf, Wf, bf,
        Uo, Wo, bo, Ug, Wg, bg
    ):
        """
        x_t.shape = (timestep=1, dim)
        x_t.shape = (n_samples, timestep=1, dim)
        """
        i_t = T.nnet.sigmoid(T.dot(x_t, Ui) + T.dot(h_tm1, Wi) + bi)
        f_t = T.nnet.sigmoid(T.dot(x_t, Uf) + T.dot(h_tm1, Wf) + bf)
        o_t = T.nnet.sigmoid(T.dot(x_t, Uo) + T.dot(h_tm1, Wo) + bo)
        g_t = T.tanh(T.dot(x_t, Ug) + T.dot(h_tm1, Wg) + bg)

        c_t = c_tm1 * f_t + g_t * i_t
        h_t = T.tanh(c_t) * o_t

        return h_t, c_t

    def forward(self, X):
        """
        X.shape = (timesteps, dim)
        X.shape = (n_samples, timesteps, dim)
        """
        states, updates = theano.scan(
            fn=self.step,
            sequences=[X],
            outputs_info=[self.h_tm1, self.c_tm1],
            non_sequences=[
                self.Ui, self.Wi, self.bi,
                self.Uf, self.Wf, self.bf,
                self.Uo, self.Wo, self.bo,
                self.Ug, self.Wg, self.bg
            ]
        )
        updates = [(self.h_tm1, states[0][-1]), (self.c_tm1, states[1][-1])]
        return states, updates


class LSTMEncoder(LSTM):
    def encode(self, X):
        states, updates = self.forward(X)
        h_t = states[0][-1]
        c_t = states[1][-1]
        return h_t, c_t, updates


class LSTMDecoder(LSTM):
    def __init__(self, size, dim, h_tm1=None, c_tm1=None):
        super(LSTMDecoder, self).__init__(size=size, dim=dim)
        self.Wh = init_weights((size, dim), "Wh")
        self.bh = shared_zeros((1, dim), "bh")

        self.h_tm1 = h_tm1 or shared_zeros((1, size), "h_tm1")
        self.c_tm1 = c_tm1 or shared_zeros((1, size), "c_tm1")

        self.y_t = shared_zeros((1, dim), "y_t")

        # self.decode_length = theano.shared(decode_length)

        self.params.append(self.Wh)
        self.params.append(self.bh)

    def decode_step(
        self, y_t, h_tm1, c_tm1,
        Ui, Wi, bi, Uf, Wf, bf,
        Uo, Wo, bo, Ug, Wg, bg,
        Wh, bh
    ):
        h_t, c_t = self.step(
            y_t, h_tm1, c_tm1,
            Ui, Wi, bi, Uf, Wf, bf,
            Uo, Wo, bo, Ug, Wg, bg
        )
        y_t = T.dot(h_t, Wh) + bh
        return y_t, h_t, c_t

    def decode(self, h_tm1, c_tm1, timesteps):
        outputs, updates = theano.scan(
            fn=self.decode_step,
            outputs_info=[self.y_t, h_tm1, c_tm1],
            non_sequences=[
                self.Ui, self.Wi, self.bi,
                self.Uf, self.Wf, self.bf,
                self.Uo, self.Wo, self.bo,
                self.Ug, self.Wg, self.bg,
                self.Wh, self.bh
            ],
            n_steps=timesteps
        )
        updates = [
            (self.h_tm1, outputs[1][-1]),
            (self.c_tm1, outputs[2][-1])
        ]
        return T.flatten(outputs[0], 2), updates


class Seq2Seq(object):
    def __init__(self, size, dim):
        self.encoder = LSTMEncoder(size, dim)
        self.decoder = LSTMDecoder(size, dim)
        self.params = []
        self.params += self.encoder.params
        self.params += self.decoder.params
        self._predict = None
        self._train = None
        self._test = None

    def compile(self, loss_func, optimizer):
        seq_input = T.fmatrix()
        seq_target = T.fmatrix()
        decode_timesteps = T.iscalar()

        h_tm1, c_tm1, updates_encode = self.encoder.encode(seq_input)
        seq_predict_flex, updates_decode_flex = self.decoder.decode(h_tm1, c_tm1, decode_timesteps)
        seq_predict, updates_decode = self.decoder.decode(h_tm1, c_tm1, T.shape(seq_target)[0])

        loss = loss_func(seq_predict, seq_target)
        self._predict = theano.function([seq_input, decode_timesteps], seq_predict_flex, updates=updates_encode+updates_decode_flex)
        self._test = theano.function([seq_input, seq_target], loss, updates=updates_encode+updates_decode)
        updates = []
        updates += updates_encode
        updates += updates_decode
        updates += optimizer(self.params, loss)
        self._train = theano.function([seq_input, seq_target], loss, updates=updates)

    def predict(self, seq_input, decode_timesteps):
        self.encoder.reset_state()
        self.decoder.reset_state()
        return self._predict(seq_input, decode_timesteps)

    def train(self, seq_input, seq_target):
        self.encoder.reset_state()
        self.decoder.reset_state()
        return self._train(seq_input, seq_target)

    def test(self, seq_input, seq_target):
        self.encoder.reset_state()
        self.decoder.reset_state()
        return self._test(seq_input, seq_target)


if __name__ == "__main__":
    size = 10
    dim = 4
    input_timesteps = 10
    output_timesteps = 10

    seq2seq = Seq2Seq(size, dim)
    seq2seq.compile(loss_func=categorical_crossentropy, optimizer=adadelta)

    si = np.random.rand(input_timesteps, dim).astype(dtype)
    st = np.random.rand(output_timesteps, dim).astype(dtype)

    print(seq2seq.train(si, st))
    so = seq2seq.predict(si, output_timesteps)

    print(so)
    loss = seq2seq.test(si, so)

    print(loss)
