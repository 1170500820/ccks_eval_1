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

