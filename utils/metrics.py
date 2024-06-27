import numpy as np

# Last step


def last_step_indices(info):
    tgtNumber = info[:, 2]

    tgtNumberChange = tgtNumber[1:] != tgtNumber[:-1]
    return np.where(tgtNumberChange)[0]


def last_step_distances_nparray(x, y, info):
    last_indices = last_step_indices(info)

    hand = y[last_indices]
    target = x[last_indices, 2:7]

    distances = np.sqrt(np.mean(np.square(hand - target), axis=1))

    return distances


def last_step_distances(x, y, info):
    if isinstance(x, list) and isinstance(y, list) and isinstance(info, list):
        return [
            last_step_distances_nparray(x_, y_, info_)
            for x_, y_, info_ in zip(x, y, info)
        ]

    if (
        isinstance(x, np.array)
        and isinstance(y, np.array)
        and isinstance(info, np.array)
    ):
        return last_step_distances_nparray(x, y, info)


# Red steps


def red_indices(info):
    tgtRed = info[:, 3]

    red_indices = np.where(tgtRed)[0]
    return red_indices


def red_distances_nparray(x, y, info):
    last_indices = red_indices(info)

    hand = y[last_indices]
    target = x[last_indices, 2:7]

    distances = np.sqrt(np.mean(np.square(hand - target), axis=1))

    return distances


def red_distances(x, y, info):
    if isinstance(x, list) and isinstance(y, list) and isinstance(info, list):
        return [
            red_distances_nparray(x_, y_, info_) for x_, y_, info_ in zip(x, y, info)
        ]

    if (
        isinstance(x, np.array)
        and isinstance(y, np.array)
        and isinstance(info, np.array)
    ):
        return red_distances_nparray(x, y, info)


# All distances


def all_distances_nparray(x, test, pred):
    hand = test
    model = pred
    target = x[:, 2:7]

    hand_model = np.sqrt(np.mean(np.square(hand - model), axis=1))
    target_model = np.sqrt(np.mean(np.square(target - model), axis=1))

    return hand_model, target_model


def all_distances(x, test, pred):
    if isinstance(x, list) and isinstance(test, list) and isinstance(pred, list):
        return [
            all_distances_nparray(x_, test_, pred_)
            for x_, test_, pred_ in zip(x, test, pred)
        ]

    if (
        isinstance(x, np.array)
        and isinstance(test, np.array)
        and isinstance(pred, np.array)
    ):
        return all_distances(x, test, pred)
