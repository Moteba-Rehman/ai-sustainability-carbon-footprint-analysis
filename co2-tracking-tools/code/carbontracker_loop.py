from carbontracker.tracker import CarbonTracker

tracker = CarbonTracker(epochs=1)

tracker.epoch_start()

total = 0
for i in range(10000000):
    total += i

tracker.epoch_end()
tracker.stop()

print("Total:", total)