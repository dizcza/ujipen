import numpy as np


def l2_dist(x, y):
    axis = 1 if max(x.ndim, y.ndim) > 1 else None
    return np.sum((x - y) ** 2, axis=axis)


def dtw_vanilla(sample, pattern):
    dist_matrix = np.zeros((len(sample) + 1, len(pattern) + 1))
    dist_matrix[0, 1:] = np.inf
    dist_matrix[1:, 0] = np.inf
    for i, s in enumerate(sample, start=1):
        for j, p in enumerate(pattern, start=1):
            cost = l2_dist(s, p)
            insertion = dist_matrix[i - 1, j]
            deletion = dist_matrix[i, j - 1]
            match = dist_matrix[i - 1, j - 1]
            dist_matrix[i, j] = cost + min(insertion, deletion, match)
    dist_matrix = dist_matrix[1:, 1:]
    return dist_matrix


def dtw_online(sample, pattern):
    n = len(pattern) + 2
    dist_top = np.repeat(np.inf, n)
    dist_top[1] = 0
    k = 0  # declare k with any value
    for i, s in enumerate(sample, start=1):
        for j, p in enumerate(pattern, start=1):
            cost = l2_dist(s, p)
            k = (j - i + 1) % n
            insertion = dist_top[(k + 1) % n]
            deletion = dist_top[(k - 1) % n]
            match = dist_top[k]
            dist_top[k] = cost + min(insertion, deletion, match)
        dist_top[(k + 1) % n] = np.inf
        assert np.isinf(dist_top).sum() == 2
    dist_answer = dist_top[k]
    return dist_answer


def dtw_vectorized(sample, pattern):
    sample_expanded = np.expand_dims(sample, axis=1)  # (n_s, 1, 2)
    pattern_expanded = np.expand_dims(pattern, axis=0)  # (1, n_p, 2)
    D = np.empty((len(sample) + 1, len(pattern) + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    D[0, 0] = 0
    D[1:, 1:] = np.sum((sample_expanded - pattern_expanded) ** 2, axis=2)

    r, c = np.array(D.shape) - 1
    for a in range(1, r + c):
        # We have I>=0, I<r, J>0, J<c and J-I+1=a
        I = np.arange(max(0, a - c), min(r, a))
        J = I[::-1] + a - min(r, a) - max(0, a - c)
        # We have to use two np.minimum because np.minimum takes only two args.
        D[I + 1, J + 1] += np.minimum(np.minimum(D[I, J], D[I, J + 1]), D[I + 1, J])

    return D[1:, 1:]
