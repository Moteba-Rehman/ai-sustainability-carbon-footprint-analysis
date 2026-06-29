from codecarbon import EmissionsTracker
import numpy as np

tracker = EmissionsTracker()
tracker.start()

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

C = np.dot(A, B)

emissions = tracker.stop()

print("Matrix multiplication completed.")
print("Emissions:", emissions, "kg CO2eq")