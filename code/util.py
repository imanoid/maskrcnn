from datetime import datetime, timedelta

class StopWatch(object):
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = datetime.now()

    def stop(self):
        return (datetime.now() - self.start_time).total_seconds() * 1e3