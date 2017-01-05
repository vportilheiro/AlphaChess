###
#
#   trainer.py
#
#   File responsible for machine learning of chess agent.
#
#   Vasco Portilheiro, 2016
###

import chess, chess.pgn
import controller
import agents

import numpy as np
import cPickle
import gzip
import glob
import os
import random

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils

# Will preprocess the given pgn data file, writing training data files
# in 'data/', each with [gamesPerFile] games. The output filenames have
# the format '[outFilePrefix]_[number]'.
# If [ngames] is provided, only that number of games will be preprocessed.
# [startAt] allows one to skip some number of games in the output file
# before starting preprocessing.
def preprocess(dataFile, outFilePrefix, gamesPerFile=5000, ngames=0, startAt=0, onlyWins=False):

    # Seek forward in file until given starting game
    pgn = open(dataFile)
    for i in xrange(startAt):
        chess.pgn.read_game(pgn)

    # Inputs and labeled output for each network
    psInputs, psLabels = [], []     # piece selector
    pmInputs, pmLabels = [], []     # pawn move
    knmInputs, knmLabels = [], []   # knight move
    bmInputs, bmLabels = [], []     # bishope move
    rmInputs, rmLabels = [], []     # rook move
    qmInputs, qmLabels = [], []     # queen move
    kmInputs, kmLabels = [], []     # king move

    # Writes a set of data into a numbered output file
    def writeOut(n):
        psData = psInputs, psLabels
        pmData = pmInputs, pmLabels
        knmData = knmInputs, knmLabels
        bmData = bmInputs, bmLabels
        rmData = rmInputs, rmLabels
        qmData = qmInputs, qmLabels
        kmData = kmInputs, kmLabels
        data = [psData, pmData, knmData, bmData, rmData, qmData, kmData]
        f = gzip.open(outFilePrefix + '_' + str(n/gamesPerFile), 'wb')
        cPickle.dump(data, f, protocol=2)
        f.close()

    n = 0           # Counts number of games processed
    reset = False   # True when a new file/data-set is started
                    #  (except on first run, when it is redundant)
    while (ngames == 0) or (n < ngames):

        if n!= 0 and n%100 == 0:
            print "Preprocessed first %s games" % n

        gameNode = chess.pgn.read_game(pgn)
        if gameNode == None:
            break

        if reset:
            psInputs, psLabels = [], []
            pmInputs, pmLabels = [], []
            knmInputs, knmLabels = [], []
            bmInputs, bmLabels = [], []
            rmInputs, rmLabels = [], []
            qmInputs, qmLabels = [], []
            kmInputs, kmLabels = [], []
            reset = False

        if gameNode.headers['Result'] == '1-0':
            winner = chess.WHITE
        else:
            winner = chess.BLACK

        while not gameNode.is_end():
            board = gameNode.board()
            move = gameNode.variations[0].move

            # If onlyWins is set, only process turns from player
            # who won the current game
            if (not onlyWins) or (board.turn == winner):
                boardTensor = boardToTensor(board, board.turn)
                psInputs.append(boardTensor)
                piece = board.piece_at(move.from_square)
                pieceSelectorBoard = chess.BaseBoard('8/8/8/8/8/8/8/8')
                pieceSelectorBoard.set_piece_at(move.from_square, piece)
                psLabels.append(boardToProbability(pieceSelectorBoard, board.turn, piece.piece_type))
                
                moveSelectorBoard = chess.BaseBoard('8/8/8/8/8/8/8/8')
                moveSelectorBoard.set_piece_at(move.to_square, piece)
                if piece.piece_type == chess.PAWN:
                    pmInputs.append(boardTensor)
                    pmLabels.append(boardToProbability(moveSelectorBoard, board.turn, piece.piece_type))
                elif piece.piece_type == chess.KNIGHT:
                    knmInputs.append(boardTensor)
                    knmLabels.append(boardToProbability(moveSelectorBoard, board.turn, piece.piece_type))
                elif piece.piece_type == chess.BISHOP:
                    bmInputs.append(boardTensor)
                    bmLabels.append(boardToProbability(moveSelectorBoard, board.turn, piece.piece_type))
                elif piece.piece_type == chess.ROOK:
                    rmInputs.append(boardTensor)
                    rmLabels.append(boardToProbability(moveSelectorBoard, board.turn, piece.piece_type))
                elif piece.piece_type == chess.QUEEN:
                    qmInputs.append(boardTensor)
                    qmLabels.append(boardToProbability(moveSelectorBoard, board.turn, piece.piece_type))
                elif piece.piece_type == chess.KING:
                    kmInputs.append(boardTensor)
                    kmLabels.append(boardToProbability(moveSelectorBoard, board.turn, piece.piece_type))
            gameNode = gameNode.variations[0]
        n += 1
        if n % gamesPerFile == 0:
            writeOut(n)
            reset = True
    if not reset: writeOut(n)


# Helper function that flips iteration over ranks
# given the color black
def iterRanks(color):
    if color == chess.WHITE:
        return xrange(0,8)
    else:
        return reversed(xrange(0, 8))

# Turns board into 6x8x8 tensor, for given color.
# The color parameter is necessary to make it possible
# to train for either color, by completely inverting the
# board to be from the given players perspective.
def boardToTensor(board, color):
    result = np.zeros(shape=(6,8,8))
    for fileIndex in xrange(0,8):
        for rank, trueRank in enumerate(iterRanks(color)):
            piece = board.piece_at(chess.square(fileIndex, trueRank))
            if piece:
                # Use 1 for friendly pieces, -1 for opponent')s
                if piece.color == color:
                    result[piece.piece_type-1][fileIndex][rank] = 1
                else:
                    result[piece.piece_type-1][fileIndex][rank] = -1
    return result

# Similar to the function above, takes a board, but rather than a
# 6x8x8 tensor, outputs an 8x8 "board" with positive numbers where
# pieces of the given color and type exist
def boardToProbability(board, color, pieceType):
    result = np.zeros(shape=(8,8))
    total = 0.0
    for fileIndex in xrange(0,8):
        for rank, trueRank in enumerate(iterRanks(color)):
            piece = board.piece_at(chess.square(fileIndex, trueRank))
            if piece and piece.color == color and piece.piece_type == pieceType:
                total += 1.0
                result[fileIndex][rank] = 1
    #if total != 0:
    #   result /= total
    return result.flatten()

def createModel(optimizer='rmsprop', learningRate=0.001, decay=0):
    model = Sequential()
    model.add(Convolution2D(32, 4, 4, border_mode='same', init='uniform',
                                input_shape=(6,8,8), activation='relu'))
    model.add(Flatten(input_shape=(6,8,8)))
    model.add(Dense(128, init='uniform', bias=True, activation='relu'))
    model.add(Dense(64, init='uniform', bias=True, activation='softmax'))

    if optimizer == 'rmsprop':
        opt = RMSprop(lr=learningRate, decay=decay)
    elif optimizer == 'sgd':
        opt = SGD(lr=learningRate, decay=decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])
    return model

def createModels():
    pieceSelectorModel = createModel()
    pawnMoveModel = createModel()
    knightMoveModel = createModel()
    bishopMoveModel = createModel()
    rookMoveModel = createModel()
    queenMoveModel = createModel()
    kingMoveModel = createModel()
    return (pieceSelectorModel, pawnMoveModel, knightMoveModel,
            bishopMoveModel, rookMoveModel, queenMoveModel, kingMoveModel)

modelPrefixes = ['pieceSelector', 'pawnMove', 'knightMove', 'bishopMove',
        'rookMove', 'queenMove', 'kingMove']

# Loads a set of models. If random is false, loads the
# latest available one. Returns a tuples, where the first
# entry is the list of models, and the second is a list of
# their respective training epochs
def loadModelEpochs(modelDir='models', random=False):
    models = []
    epochs = []
    for p in modelPrefixes:
        modelRegex = os.path.join(modelDir, p + '*.h5')
        lastModel = max(glob.iglob(modelRegex))
        maxModelNum = int(lastModel[lastModel.rindex('_')+1:-3])
        if random:
            modelNum = random.randrange(0, maxModelNum+1)
        else:
            modelNum = maxModelNum
        modelFile = os.path.join(modelDir, p + '_' + str(modelNum).zfill(2) + '.h5')
        models += [load_model(modelFile)]
        epochs += [modelNum]
    return models, epochs

# Saves models and updates maxModelNum
def saveModels(models, modelNum):
    for i, model in enumerate(models):
        model.save(os.path.join('models', modelPrefixes[i] + '_' + str(modelNum) + '.h5'))
    with open(os.path.join('models', 'info.txt'), 'w+') as f:
        f.write(str(modelNum))

def dataGen(dataPrefix):
    n = 1
    while True:
        # get file string
        fileName = dataPrefix + str(n)
        # try load file, else break
        # yield
        n += 1


# Takes a two lists [models] and [epochs], as returned from loadModelEpochs,
# and trains them on the given data.
def train(models, epochs, data=None):

    if data == None:
        f = gzip.open('data10k')
        print "Loading data..."
        data = cPickle.load(f)
        print "Done!"

    # Load latest models and set corresponding starting epoch 

    hists = []
    for i,model in enumerate(models):
        print modelPrefixes[i]
        x = np.array(data[i][0])
        y = np.asarray(data[i][1])

        checkpoints = 'models_cat_opt/' + modelPrefixes[i] + '_{epoch:02d}.h5'
        checkpointer = ModelCheckpoint(checkpoints, verbose=1, save_best_only=True)
        hist = model.fit(x, y, verbose=1, validation_split=0.2, callbacks=[checkpointer], nb_epoch=10, batch_size=250)
        print(hist.history)
        hists += [hist]

    return hists

def gridSearch(data):

    f = open('grid-search', 'w')

    learningRate = [0.001, 0.0015, 0.005]
    decay = [0, 0.2, 0.6, 0.8]
    param_grid = dict(learningRate=learningRate, decay=decay)
    for i,modelName in enumerate(modelPrefixes):
        print "===== %s =====" % modelName
        model = KerasClassifier(build_fn=createModel, verbose=1, optimizer='rmsprop')
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

        grid_result = grid.fit(np.array(data[i][0]), np.asarray(data[i][1]))

        s = "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)
        print(s)
        f.write(s)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            s = "%f (%f) with: %r" % (mean, stdev, param)
            print s
            f.write(s)
    f.close()
