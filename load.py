import numpy
import theano
import theano.tensor as T

rng = numpy.random


def float_x(x):
    return numpy.asarray(x, dtype=theano.config.floatX)


def w_values(n_in, n_out):
    return numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)), dtype=theano.config.floatX)


def get_parameters(n_in=None, n_hidden_units=2048, n_hidden_layers=None, ws=None, bs=None):
    if ws is None or bs is None:
        print('initializing Ws & bs')
        if type(n_hidden_units) != list:
            n_hidden_units = [n_hidden_units] * n_hidden_layers
        else:
            n_hidden_layers = len(n_hidden_units)

        ws = []
        bs = []

        for i in range(n_hidden_layers):
            if i == 0:
                n_in_2 = n_in
            else:
                n_in_2 = n_hidden_units[i - 1]
            if i < n_hidden_layers - 1:
                n_out_2 = n_hidden_units[i]
                w = w_values(n_in_2, n_out_2)
                gamma = 0.1  # initialize it to slightly positive so the derivative exists
                b = numpy.ones(n_out_2, dtype=theano.config.floatX) * gamma
            else:
                w = numpy.zeros(n_in_2, dtype=theano.config.floatX)
                b = float_x(0.)
            ws.append(w)
            bs.append(b)

    ws_s = [theano.shared(W) for W in ws]
    bs_s = [theano.shared(b) for b in bs]

    return ws_s, bs_s


def get_model(ws_s, bs_s, dropout=False):
    print('building expression graph')
    x_s = T.matrix('x')

    if type(dropout) != list:
        dropout = [dropout] * len(ws_s)

    # Convert input into a 12 * 64 list
    pieces = []
    for piece in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
        # pieces.append((x_s <= piece and x_s >= piece).astype(theano.config.floatX))
        pieces.append(T.eq(x_s, piece))

    binary_layer = T.concatenate(pieces, axis=1)

    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))

    last_layer = binary_layer
    n = len(ws_s)
    for l in range(n - 1):
        # h = T.tanh(T.dot(last_layer, Ws[l]) + bs[l])
        h = T.dot(last_layer, ws_s[l]) + bs_s[l]
        h = h * (h > 0)

        if dropout[l]:
            mask = srng.binomial(n=1, p=0.5, size=h.shape)
            h = h * T.cast(mask, theano.config.floatX) * 2

        last_layer = h

    p_s = T.dot(last_layer, ws_s[-1]) + bs_s[-1]
    return x_s, p_s
