from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

total = 0
for i in range(10000000):
    total += i

emissions = tracker.stop()

print("Total:", total)
print("Emissions:", emissions, "kg CO2eq")