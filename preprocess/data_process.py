import random


def randomize(*lsts):
    """
    randomize([1,2,3], [4,5,6])
    打乱顺序
    :param lsts: iterables of iterables,
    :return:
    """
    zipped = list(zip(*lsts))
    random.shuffle(zipped)
    return list(zip(*zipped))


def train_val_split(lst, split_ratio=None):
    cnt = len(lst)
    bound = int(cnt * split_ratio)
    return lst[:bound], lst[bound: ]


def multi_train_val_split(*lst, split_ratio=None):
    """

    :param lst: [lst1, lst2, ...]
    :param split_ratio:
    :return: train_lst1
    """
    trains, vals = [], []
    for l in lst:
        train_l, val_l = train_val_split(l, split_ratio=split_ratio)
        trains.append(train_l)
        vals.append(val_l)
    return trains + vals
