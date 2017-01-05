###
#
#   agents.py
#
#   This file contains the definition for chess agents,
# all of which instantiate the abstract Agent class. This
# class defines one method, getMove(board), which returns
# a move for the given board position.
#   
#   Vasco Portilheiro, 2016
###

import chess, chess.uci
import numpy as np
from subprocess import check_output

# Abstract agent class: defines method to make move from a position
class Agent:
    
    # Returns the move for current player to
    # make from the given [board] position,
    # guarranteed to be valid
    def getMove(self, board): raise NotImplementedError("Override me")

# A human agent enters the move to play at the command line
class HumanAgent(Agent):

    def getMove(self, board):
        if board.turn == chess.WHITE:
            prompt = "White: "
        else:
            prompt = "Black: "
        while True:
            try:
                move = board.parse_san(raw_input(prompt))
                break
            except ValueError:
                print("Invalid input! Please input a valid move in standard algebraic notation.")

        return move

# Implements a baseline chess engine to compare against
# other engines. This engine will perform a depth limited
# minimax search in order to choose the next move.
class SimpleEngine(Agent):

    # Constructor allows setting of maximum depth of search,
    # as well as values to give to pieces during evaluation,
    # in a list [pawns, knights, bishops, rooks, queen, king].
    # A mobility weight can also be provided, which, if not None,
    # will make the engine prefer position which lead to more possible
    # moves.
    def __init__(self, depth=2, pieceValues=[1,3,3,5,9,200], memoize=True,
                mobilityWeight=0.1, winUtility=float('inf'), drawUtility=0):
        self.half_depth = 2*depth
        self.pieceValues = pieceValues
        self.mobilityWeight = mobilityWeight
        self.winUtility = winUtility
        self.drawUtility = drawUtility
        self.memoize = memoize

    # Returns an evaluation of the given board, based on the
    # formula first described by Shannon, where the score
    # is the sum of differences in number of each type of piece
    # weighed differently for each piece type. Finally, the
    # value is turned negative for black, who is the minimizing
    # agent in the minimax search.
    def evaluate(self, board):
        score = 0
        pieceType = chess.PAWN
        while pieceType <= chess.KING:
            score += self.pieceValues[pieceType-1]*\
                (len(board.pieces(pieceType, chess.WHITE)) -\
                 len(board.pieces(pieceType, chess.BLACK)))
            pieceType += 1
        if self.mobilityWeight != None:
            score += self.mobilityWeight * len(board.legal_moves)
        if board.turn == chess.BLACK:
            score = -score
        return score

    # Calculates the utility of an end-state,
    # assuming that the game is over
    def utility(self, board):
        if board.result == '1-0':
            return self.winUtility
        elif board.result == '0-1':
            return -self.winUtility
        else:
            return self.drawUtility

    def minimax(self, board, half_depth, alpha, beta):
        if self.memoize:
            board_hash = board.zobrist_hash()
            if board_hash in self.memos:
                self.cache_hits += 1
                return self.memos[board_hash]
        if board.is_game_over(claim_draw=True):
            return (self.utility(board), None)
        if half_depth == 0:
            return (self.evaluate(board), None)

        if board.turn == chess.WHITE:
            minimaxVal = float('-inf')
            minimaxMove = None
            for move in board.legal_moves:
                board.push(move)
                val, nextMove = self.minimax(board, half_depth-1, alpha, beta)
                minimaxVal, minimaxMove = max((minimaxVal, minimaxMove), (val, move))
                alpha = max(alpha, minimaxVal)
                board.pop()
                if beta <= alpha: break 
        else:
            minimaxVal = float('inf')
            minimaxMove = None
            for move in board.legal_moves:
                board.push(move)
                val, nextMove = self.minimax(board, half_depth-1, alpha, beta)
                minimaxVal, minimaxMove = min((minimaxVal, minimaxMove), (val, move))
                beta = min(beta, minimaxVal)
                board.pop()
                if beta <= alpha: break 
        if self.memoize:
            self.memos[board_hash] = (minimaxVal, minimaxMove)
        self.moves += 1
        return minimaxVal, minimaxMove

    def getMove(self, board):
        if self.memoize:
            self.memos = {}
        self.moves = 0
        self.cache_hits = 0
        move = self.minimax(board, self.half_depth, float('-inf'), float('inf'))[1]
        print "%s moves examined\t%s cache hits" % (self.moves, self.cache_hits)
        print "AI: %s" % move
        return move

# Implementa an agent that plays based on the Stockfish engine
class StockfishAgent(Agent):

    def __init__(self, moveTime=500):
        enginePath  = check_output(['which', 'stockfish']).strip()
        self.engine = chess.uci.popen_engine(enginePath)
        self.engine.uci()
        self.moveTime = moveTime

    def getMove(self, board):
        self.engine.position(board)
        return self.engine.go(movetime=self.moveTime).bestmove

class PolicyAgent(Agent):

    def __init__(self, models):
        enginePath  = check_output(['which', 'stockfish']).strip()
        self.engine = chess.uci.popen_engine(enginePath)
        self.engine.uci()
        self.models = models
        self.pieceSelector = models[0]

    def getMove(self, board):
        self.engine.position(board)

        t = trainer.boardToTensor(board, board.turn)
        startProbs = self.pieceSelector.predict(np.array([t])).reshape(64)
        maxProb = -1
        maxStartIdx = -1
        maxEndIdx = -1
        for i, sourceProb in enumerate(startProbs):
            startIdx = i
            endProb = self.models[board.piece_type_at(i)].predict(np.array([t])).reshape(64)
            probIndex = [(p,i) for (i,p) in enumerate(endProb)]
            endProb, endIdx = max(probIndex)
            prob = sourceProb * endP
            if prob > maxProb:
                maxProb = prob
                maxStartIdx = startIdx
                maxEndIdx = endIdx
        # Flip board if black -- due to models being trained
        # always from 'friendly'/own side
        if board.turn == chess.BLACK:

