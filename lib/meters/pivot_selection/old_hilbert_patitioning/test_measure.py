import time
import numpy as np

from measure import Measure


def test_measure():
    def test(counter):
        for _ in range(10):
            counter()
        time.sleep(0.05)

    with Measure() as m:
        test(m.count_event)
        test(m.count_event)

    assert m.event_count == 2 * 10
    assert np.allclose(m.execution_time, 0.05 * 2, atol=0.005)
