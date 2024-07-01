import timeit


class Measure:
    def __init__(self):
        self.event_count = 0
        self.execution_time = 0

    def count_event(self):
        self.event_count += 1

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = timeit.default_timer()
        self.execution_time = self.end_time - self.start_time

    def get_results(self):
        return self.execution_time, self.event_count
