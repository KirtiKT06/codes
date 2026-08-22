import random

# Enter 8 names
names = ["A", "Sachi", "Manas", "Ranjitha"]

# Shuffle the names
random.shuffle(names)

# Create pairs
pairs = [(names[i], names[i + 1]) for i in range(0, len(names), 2)]

# Display pairs
print("Random Pairs:")
for pair_num, pair in enumerate(pairs, start=1):
    print(f"Pair {pair_num}: {pair[0]} & {pair[1]}")