"""The toy model of src/main.vx, in Python, printing the tokens it decodes.

A third implementation that shares no code with the Vx one: build.sh compares its tokens with
the ones the GPU + NPU run printed. Every number in the model is a whole number small enough
that f32 (in Vx) and Python's floats give the same results.
"""


def embed(t, i):
    return ((t >> i) & 1) * 2 - 2 + t % (i + 2)


def main():
    table = {t: [embed(t, i) for i in range(4)] for t in range(16)}
    cache = [table[(p * 5 + 3) % 16] for p in range(8)]
    out = []
    for _ in range(8):
        last = cache[-1]
        mix = [0, 0, 0, 0]
        for row in cache:
            s = sum(last[i] * row[i] for i in range(4))
            w = (s + 4) ** 2
            for i in range(4):
                mix[i] += w * row[i]
        best, best_score = -1, None
        for t in range(16):
            if table[t] in cache:
                continue
            score = sum(table[t][i] * mix[i] for i in range(4))
            if best < 0 or score > best_score:
                best, best_score = t, score
        out.append(best)
        cache.append(table[best])
    print(" ".join(str(t) for t in out))


main()
