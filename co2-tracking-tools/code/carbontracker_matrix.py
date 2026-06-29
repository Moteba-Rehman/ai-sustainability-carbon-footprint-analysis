from carbontracker.tracker import CarbonTracker
import numpy as np

tracker = CarbonTracker(epochs=1)

tracker.epoch_start()

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

C = np.dot(A, B)

tracker.epoch_end()
tracker.stop()

print("Matrix multiplication completed.")