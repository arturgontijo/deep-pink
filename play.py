import load
import theano
import chess, chess.pgn
import time
import re
import numpy
import sunfish
import pickle
import random
import traceback

strip_whitespace = re.compile(r"\s+")
translate_pieces = bytes.maketrans(b".pnbrqkPNBRQK",
                                   b"\x00" + b"\x01\x02\x03\x04\x05\x06" + b"\x08\x09\x0a\x0b\x0c\x0d")
CHECKMATE_SCORE = 1e6


def get_model_from_pickle(fn):
    f = open(fn, "rb")
    ws, bs = pickle.load(f, encoding='latin1')

    ws_s, bs_s = load.get_parameters(ws=ws, bs=bs)
    x, p = load.get_model(ws_s, bs_s)

    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict


def sf2array(pos, flip):
    # Create a numpy array from a sunfish representation
    pos = strip_whitespace.sub('', pos.board)  # should be 64 characters now
    pos = pos.translate(translate_pieces)
    m = numpy.fromstring(pos, dtype=numpy.int8)
    if flip:
        m = numpy.fliplr(m.reshape(8, 8)).reshape(64)
    return m


def negamax(pos, depth, alpha, beta, color, func):
    moves = []
    x = []
    pos_children = []
    for move in pos.gen_moves():
        pos_child = pos.move(move)
        moves.append(move)
        x.append(sf2array(pos_child, flip=(color == 1)))
        pos_children.append(pos_child)

    if len(x) == 0:
        return Exception('eh?')

    # Use model to predict scores
    scores = func(x)

    for i, pos_child in enumerate(pos_children):
        if pos_child.board.find('K') == -1:
            scores[i] = CHECKMATE_SCORE

    child_nodes = sorted(zip(scores, moves), reverse=True)

    best_value = float('-inf')
    best_move = None

    for score, move in child_nodes:
        if depth == 1 or score == CHECKMATE_SCORE:
            value = score
        else:
            # print('ok will recurse', sunfish.render(move[0]) + sunfish.render(move[1])
            pos_child = pos.move(move)
            neg_value, _ = negamax(pos_child, depth - 1, -beta, -alpha, -color, func)
            value = -neg_value

        # value += random.gauss(0, 0.001)

        # crdn = sunfish.render(move[0]) + sunfish.render(move[1])
        # print('\t' * (3 - depth), crdn, score, value

        if value > best_value:
            best_value = value
            best_move = move

        if value > alpha:
            alpha = value

        if alpha > beta:
            break

    return best_value, best_move


def create_move(board, crdn):
    # workaround for pawn promotions
    move = chess.Move.from_uci(crdn)
    if board.piece_at(move.from_square).piece_type == chess.PAWN:
        if int(move.to_square / 8) in [0, 7]:
            move.promotion = chess.QUEEN  # always promote to queen
    return move


class Player(object):
    def move(self, gn_current):
        raise NotImplementedError()


class ComputerA(Player):
    def __init__(self, func, maxd=5):
        self._func = func
        self._pos = sunfish.Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)
        self._maxd = maxd

    def move(self, gn_current):
        assert (gn_current.board().turn is True)

        if gn_current.move is not None:
            # Apply last_move
            crdn = str(gn_current.move)
            move = (119 - sunfish.parse(crdn[0:2]), 119 - sunfish.parse(crdn[2:4]))
            self._pos = self._pos.move(move)

        # for depth in xrange(1, self._maxd+1):
        alpha = float('-inf')
        beta = float('inf')

        depth = self._maxd
        t0 = time.time()
        best_value, best_move = negamax(self._pos, depth, alpha, beta, 1, self._func)
        crdn = sunfish.render(best_move[0]) + sunfish.render(best_move[1])
        print(depth, best_value, crdn, time.time() - t0)

        self._pos = self._pos.move(best_move)
        crdn = sunfish.render(best_move[0]) + sunfish.render(best_move[1])
        move = create_move(gn_current.board(), crdn)

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new


class ComputerB(Player):
    def __init__(self, func, maxd=5):
        self._func = func
        self._pos = sunfish.Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)
        self._maxd = maxd

    def move(self, gn_current):
        assert (gn_current.board().turn is False)

        # Apply last_move
        crdn = str(gn_current.move)
        move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
        self._pos = self._pos.move(move)

        # for depth in xrange(1, self._maxd+1):
        alpha = float('-inf')
        beta = float('inf')

        depth = self._maxd
        t0 = time.time()
        best_value, best_move = negamax(self._pos, depth, alpha, beta, 1, self._func)
        crdn = sunfish.render(best_move[0]) + sunfish.render(best_move[1])
        print(depth, best_value, crdn, time.time() - t0)

        self._pos = self._pos.move(best_move)
        crdn = sunfish.render(119 - best_move[0]) + sunfish.render(119 - best_move[1])
        move = create_move(gn_current.board(), crdn)

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new


class Human(Player):
    @staticmethod
    def get_move(bb, move_str):
        try:
            move = chess.Move.from_uci(move_str)
        except:
            print('cant parse')
            return False
        if move not in bb.legal_moves:
            print('not a legal move')
            return False
        else:
            return move

    def move(self, gn_current):
        bb = gn_current.board()

        while True:
            move = self.get_move(bb, input('your turn: '))
            if move:
                break

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new


class Sunfish(Player):
    def __init__(self, secs=1):
        self._searcher = sunfish.Searcher()
        self._pos = sunfish.Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)
        self._secs = secs

    def move(self, gn_current):
        assert (gn_current.board().turn is False)

        # Apply last_move
        crdn = str(gn_current.move)
        move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
        self._pos = self._pos.move(move)

        t0 = time.time()
        move, score = self._searcher.search(self._pos, self._secs)
        print(time.time() - t0, move, score)
        self._pos = self._pos.move(move)

        crdn = sunfish.render(119 - move[0]) + sunfish.render(119 - move[1])
        move = create_move(gn_current.board(), crdn)

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new


def pprint_board(board):
    board = str(board)
    print("    a b c d e f g h")
    print("-------------------")
    lines = board.split("\n")
    for idx, line in enumerate(lines):
        print("{} | {}".format(8 - idx, line))


def game(func):
    maxd = random.randint(1, 2)  # max depth for deep pink
    secs = random.random()  # max seconds for sunfish

    print('maxd %f secs %f' % (maxd, secs))

    opt_player = input("1=Player or 2=Spectator? ")
    if opt_player == "2":
        player_a = ComputerA(func, maxd=maxd)
        player_b = Sunfish(secs=secs)
    else:
        player_a = Human()
        player_b = ComputerB(func, maxd=maxd)

    gn_current = chess.pgn.Game()
    pprint_board(gn_current.board())

    times = {'A': 0.0, 'B': 0.0}

    while True:
        for side, player in [('A', player_a), ('B', player_b)]:
            t0 = time.time()
            try:
                gn_current = player.move(gn_current)
            except KeyboardInterrupt:
                return
            except:
                traceback.print_exc()
                return side + '-exception', times

            times[side] += time.time() - t0
            print('=========== Player %s: %s' % (side, gn_current.move))
            pprint_board(gn_current.board())
            s = str(gn_current.board())
            if gn_current.board().is_checkmate():
                return side, times
            elif gn_current.board().is_stalemate():
                return '-', times
            elif gn_current.board().can_claim_fifty_moves():
                return '-', times
            elif s.find('K') == -1 or s.find('k') == -1:
                # Both AI's suck at checkmating, so also detect capturing the king
                return side, times


def play():
    func = get_model_from_pickle('model.pickle')
    while True:
        side, times = game(func)
        f = open('stats.txt', 'a')
        f.write('%s %f %f\n' % (side, times['A'], times['B']))
        f.close()


if __name__ == '__main__':
    play()
