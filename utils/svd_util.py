import numpy as np


def random_unit_vector(n):
    unnormalized = np.random.normal(0, 1, n)
    return unnormalized / np.linalg.norm(unnormalized)


def max_svd_unit(A, epsilon=1e-10):
    error = np.inf
    v = random_unit_vector(A.shape[1])
    while error > epsilon:
        w = A.dot(v)
        alpha = np.linalg.norm(w)
        u = w / alpha

        z = A.T.dot(u)
        sigma = np.linalg.norm(z)
        v = z / sigma

        error = np.linalg.norm(A.dot(v) - sigma * u)
    return u, sigma, v


def reduced_svd(A, epsilon=1e-10):
    current_approx = np.zeros(A.shape)
    n, m = A.shape

    U = np.zeros(A.shape)
    E = np.zeros(m)
    V_t = np.zeros((A.shape[1], A.shape[1]))

    for i in range(min(A.shape)):
        u, sigma, v = max_svd_unit(A - current_approx, epsilon)
        current_approx += sigma * np.outer(u, v)
        U[:, i] = u
        E[i] = sigma
        V_t[i, :] = v
    return U, E, V_t


def restore_from_reduced(u, e, v_t, k=None):
    if k is None or k >= e.shape[0]:
        return np.dot(u * e, v_t)
    A = np.zeros((u.shape[0], v_t.shape[0]))
    for i in range(k):
        A += e[i] * np.outer(u[:, i], v_t[i, :])
    return A
