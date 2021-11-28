import random


def random_swap(words, n):
    words = words.split()
    for i in range(n):
        words = swap(words)

    return " ".join(words)


def swap(words):
    idx = random.sample(range(len(words)), 2)
    print(words, idx)
    words[idx[0]], words[idx[1]] = words[idx[1]], words[idx[0]]

    return words