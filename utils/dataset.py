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

X_COLUMNS = SHOULDER + TARGET
Y_COLUMNS = HAND

subject_ids = list(data.keys())

X = [data[subject][X_COLUMNS].to_numpy() for subject in subject_ids]
Y = [data[subject][Y_COLUMNS].to_numpy() for subject in subject_ids]


def kfold(x=None, y=None):
    if x is None or y is None:
        x, y = X, Y

    kfold = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kfold.split(x, y)):
        xtrain = [X[idx] for idx in train_index]
        ytrain = [Y[idx] for idx in train_index]
        xtest = [X[idx] for idx in test_index]
        ytest = [Y[idx] for idx in test_index]

        yield xtrain, ytrain, xtest, ytest
