from datetime import datetime, timedelta
import typing
import random


class StopWatch(object):
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = datetime.now()

    def stop(self):
        return (datetime.now() - self.start_time).total_seconds() * 1e3


def is_iterable(o):
    try:
        iter(o)
        return True
    except:
        return False


def get_time_string():
    now = datetime.now()
    return "{}.{}.{}-{}.{}.{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def random_subset(elements: typing.List, n: int):
    return [elements[i] for i in random.sample(range(len(elements)), n)]
