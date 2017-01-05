###
#
#   controller.py
#
#   The controller can be used to play/simulate games
# between two agents, as well as evaluate an agent against
# some another (where the later is the "oracle").
#
#   Vasco Portilheiro, 2016
###

import agents
import chess
import sys
import time

# Given two Agent instances, will alternate
# querying each for a move, until the game is
# over. Unless quiet is set, will print the board
# and move made after each move. If quiet is not set,
# may also be told to ;printMoveTime'.
def play(white, black, quiet=False, printMoveTime=False):
    board = chess.Board()
    currentAgent = white
    while True:
        # Print board and get move
        if not quiet:
            print(board)
            if printMoveTime:
                startTime = time.time()
        move = currentAgent.getMove(board)
        if not quiet and printMoveTime:
            endTime = time.time()
            print "%s seconds used for move" % (endTime - startTime)
        board.push(move)
        # Switch players
        if currentAgent == white: currentAgent = black
        else: currentAgent = white
        # Check for game-over, including three-fold repetition
        # and fifty-move rule draws
        if board.is_game_over(claim_draw=True):
            break
        if not quiet:
            sys.stdout.flush()
    if not quiet:
        print(board)
        print "Game over!"
    if board.result() == '1-0':
        result = [1,0]
        if not quiet:
            print "White wins!"
    elif board.result() == '0-1':
        result = [0,1]
        if not quiet:
            print "Black wins!"
    else:
        result = [0.5,0.5]
        if not quiet:
            print "It's a draw!"
    return result

# Will run nGames games of the given agent against a
# Stockfish engine based agent, and return a tuple
# of (wins, totalGames). Will alternate games where the
# given agent plays as white/black.
def evaluateAgent(agent, nGames=200, moveTime=500):
    wins = 0
    stockfishAgent = agents.StockfishAgent(moveTime = moveTime)
    for i in xrange(nGames):
        if i % 2 == 0:
            result = play(agent, stockfishAgent, quiet=True)
            wins += result[0]
        else:
            result = play(stockfishAgent, agent, quiet=True)
            wins += results[1]
    return (wins, nGames)

def main():
    ai, aiMem = agents.SimpleEngine(memoize=False), agents.SimpleEngine()
    play(ai, aiMem)

if __name__ == '__main__':
    main()
