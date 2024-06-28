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


def test_measure():
    import time
    import numpy as np

    def test(counter):
        for _ in range(10):
            counter()
        time.sleep(0.05)

    with Measure() as m:
        test(m.count_event)
        test(m.count_event)

    assert m.event_count == 2 * 10
    assert np.allclose(m.execution_time, 0.05 * 2, atol=0.005)
