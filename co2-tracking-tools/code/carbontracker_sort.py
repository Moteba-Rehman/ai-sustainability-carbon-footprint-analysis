from carbontracker.tracker import CarbonTracker
import random

tracker = CarbonTracker(epochs=1)

tracker.epoch_start()

arr = [random.randint(1, 1000000) for _ in range(1000000)]
arr.sort()

tracker.epoch_end()
tracker.stop()

print("Sorting completed.")