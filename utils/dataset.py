import pickle
from sklearn.model_selection import KFold

# Loading pre-processed data
with open("./data/preprocessed_subjectwise2.pickle", "rb") as file:
    data = pickle.load(file)

# Set definition
SHOULDER = [
    "shoulderPitch",
    "shoulderRoll",
]
HAND = [
    "endEffX",
    "endEffY",
    "endEffZ",
    "endEffPitch",
    "endEffRoll",
]
TARGET = [
    "tgtX",
    "tgtY",
    "tgtZ",
    "tgtPitch",
    "tgtRoll",
]
ARM = [
    "armYaw",
    "elbowPitch",
    "forearmYaw",
    "wristPitch",
    "wristRoll",
]
ADDITIONAL_INFO = [
    "timestamp",
    "subjectId",
    "tgtNumber",
    "tgtRed",
]

X_COLUMNS = SHOULDER + TARGET
Y_COLUMNS = HAND

subject_ids = list(data.keys())

X = [data[subject][X_COLUMNS].to_numpy() for subject in subject_ids]
Y = [data[subject][Y_COLUMNS].to_numpy() for subject in subject_ids]
INFO = [data[subject][ADDITIONAL_INFO].to_numpy() for subject in subject_ids]


def kfold(x=None, y=None):
    if x is None:
        x = X
    if y is None:
        y = Y

    kfold = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kfold.split(x, y)):
        xtrain = [X[idx] for idx in train_index]
        ytrain = [Y[idx] for idx in train_index]
        xtest = [X[idx] for idx in test_index]
        ytest = [Y[idx] for idx in test_index]

        yield xtrain, ytrain, xtest, ytest


def kfold_infos(x=None, y=None, info=None):
    if x is None:
        x = X
    if y is None:
        y = Y
    if info is None:
        info = INFO

    kfold = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kfold.split(x, y)):
        xtrain = [X[idx] for idx in train_index]
        ytrain = [Y[idx] for idx in train_index]
        infotrain = [INFO[idx] for idx in train_index]
        xtest = [X[idx] for idx in test_index]
        ytest = [Y[idx] for idx in test_index]
        infotest = [INFO[idx] for idx in test_index]

        yield xtrain, ytrain, infotrain, xtest, ytest, infotest
