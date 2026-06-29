from codecarbon import EmissionsTracker
import random

tracker = EmissionsTracker()
tracker.start()

arr = [random.randint(1, 1000000) for _ in range(1000000)]
arr.sort()

emissions = tracker.stop()

print("Sorting completed.")
print("Emissions:", emissions, "kg CO2eq")