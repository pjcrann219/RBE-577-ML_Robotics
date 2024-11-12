import random


def randomize_data(name, entries, split):
    val_file = open(f"validation_{name}.txt", "w")
    test_file = open(f"test_{name}.txt", "w")
    values = list(range(1, entries))
    random.shuffle(values)
    split_index = int(entries * split)
    val_values = values[:split_index]
    test_values = values[split_index:]

    for i in val_values:
        val_file.write(f"{i:05d}.jpg\n")  # Writing validation set


    for i in test_values:
        test_file.write(f"{i:05d}.jpg\n")  # Writing test set
if __name__ == "__main__":
    
    randomize_data("height20m", 2999, 0.8)
    randomize_data("height80m", 2999, 0.8)
