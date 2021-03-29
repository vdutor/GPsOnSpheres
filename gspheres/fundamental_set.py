from pathlib import Path

import numpy as np
from scipy import linalg, optimize
from scipy.special import comb as combinations, gegenbauer as ScipyGegenbauer


class FundamentalSystemCache:
    """A simple cache object to access precomputed fundamental system.

    Fundamental system are sets of points that allow the user to evaluate the spherical
    harmonics in an arbitrary dimension"""

    def __init__(self, dimension: int, load_dir="fundamental_system"):
        self.load_dir = Path(__file__).parents[0] / load_dir
        self.file_name = self.load_dir / f"fs_{dimension}D.npz"
        self.dimension = dimension

        if self.file_name.exists():
            with np.load(self.file_name) as data:
                self.cache = {k: v for (k, v) in data.items()}
        else:
            self.cache = {}

    def cache_key(self, degree: int) -> str:
        """Return the key used in the cache"""
        return f"degree_{degree}"

    def load(self, degree: int) -> np.array:
        """Load or calculate the set for given degree"""
        key = self.cache_key(degree)
        if key not in self.cache:
            print("WARNING: Cache miss - calculating  system")
            self.cache[key] = self.calculate(self.dimension, degree)
        return self.cache[key]

    def regenerate_and_save_cache(self, max_degrees: int) -> None:
        """Regenerate and overwrite saved cache to disk"""
        system = {}
        for degree in range(max_degrees):
            print(f"finding level {degree}/{max_degrees} in {self.dimension}D")
            d_system = self.calculate(
                self.dimension, degree, gtol=1e-8, num_restarts=10
            )
            system[f"degree_{degree}"] = d_system
        with open(self.file_name, "wb+") as f:
            np.savez(f, **system)

    @staticmethod
    def calculate(
        dimension: int, degree: int, *, gtol: float = 1e-5, num_restarts: int = 1
    ) -> np.array:
        return build_fundamental_system(
            dimension, degree, gtol=gtol, num_restarts=num_restarts
        )


def build_fundamental_system(
    dimension, degree, *, gtol=1e-5, num_restarts=1, debug=False
):
    """
    We build a fundamental system incrementally, by adding a new candidate vector each
    time and maximising the span of the space generated by these spherical harmonics.

    This can be done by greedily optimising the determinant of the gegenbauered Gram matrix.

    Based on [1, Defintion 3.1]

    [1] Approximation Theory and Harmonic Analysis on Spheres and Balls,
        Feng Dai and Yuan Xu, Chapter 1. Spherical Harmonics,
        https://arxiv.org/pdf/1304.2585.pdf
    """
    alpha = (dimension - 2) / 2.0
    gegenbauer = ScipyGegenbauer(degree, alpha)
    system_size = num_harmonics(dimension, degree)

    # 1. Choose first direction in system to be north pole
    Z0 = np.eye(dimension)[-1]
    X_system = normalize(Z0).reshape(1, dimension)
    M_system_chol = cholesky_of_gegenbauered_gram(gegenbauer, X_system)

    # 2. Find a new vector incrementally by max'ing the determinant of the gegenbauered Gram
    for i in range(1, system_size):

        Z_next, ndet, restarts = None, np.inf, 0
        while restarts <= num_restarts:
            x_init = np.random.randn(dimension)
            result = optimize.fmin_bfgs(
                f=calculate_decrement_in_determinant,
                fprime=grad_calculate_decrement_in_determinant,
                x0=x_init,
                args=(X_system, M_system_chol, gegenbauer),
                full_output=True,
                gtol=gtol,
                disp=debug,
            )

            if result[1] <= ndet:
                Z_next, ndet, *_ = result
                #  TODO: we should we break when we find the best vector.
                #  Unclear how to do this at this point.
            # Try again with new x_init
            restarts += 1
        print(
            f"det: {-ndet:11.4f}, ({i + 1:5d} of {system_size:5d}, "
            f"degree {degree}: {dimension}D)"
        )
        X_next = normalize(Z_next).reshape(1, dimension)
        X_system = np.vstack([X_system, X_next])
        M_system_chol = cholesky_of_gegenbauered_gram(gegenbauer, X_system)

    return X_system


def num_harmonics(dimension: int, degree: int) -> int:
    r"""Calculate the number of spherical harmonics of a particular degree n in
    d dimensions. Referred to as N(d, n).

    param dimension:
        S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
        For a circle d=2, for a ball d=3
    param degree: degree of the harmonic
    """
    if degree == 0:
        return 1
    else:
        comb = combinations(degree + dimension - 3, degree - 1)
        return int(np.round((2 * degree + dimension - 2) * comb / degree))


def calculate_decrement_in_determinant(Z, X_system, M_system_chol, gegenbauer):
    r"""Calculate the negative determinant.

    :param Z: is a potential vector for the next fundamental point (it will get normalized)
    :param X_system: is a matrix of existing fundamental points [num_done, D]
    :param M_system_chol: is the cholesky of the matrix M of the done points [num_done, num_done]

    :return: the negative-increment of the determinant of the matrix with Z (normalized)
     added to the done points
    """
    X = normalize(Z)
    XXd = np.dot(X_system, X)  # [num_done,]

    M_new = gegenbauer(1.0)  # X normalized so X @ X^T = 1
    M_cross = gegenbauer(XXd)

    # Determinant of M is computed efficiently making use of the Schur complement
    # M = [[ M_system_chol, M_cross], [ M_cross^T, M_new]]
    # det(M) = det(M_system_chol) * det(M_new - M_cross^T M_system_chol^{-1} M_cross )
    res = linalg.solve_triangular(M_system_chol, M_cross, trans=0, lower=True)
    return np.sum(np.square(res)) - M_new


def grad_calculate_decrement_in_determinant(Z, X_system, M_system_chol, gegenbauer):

    r"""Calculate the negative determinant.

    :param Z: is a potential vector for the next fundamental point (it will get normalized)
    :param X_system: is a matrix of existing fundamental points [num_done, D]
    :param M_system_chol: is the cholesky of the matrix M of the done points [num_done, num_done]

    """
    X = normalize(Z)
    XXd = np.dot(X_system, X)  # [num_done,]

    M_cross = gegenbauer(XXd)

    res = linalg.solve_triangular(M_system_chol, M_cross, trans=0, lower=True)
    dM_cross = 2.0 * linalg.solve_triangular(M_system_chol, res, trans=1, lower=True,)
    dXXd = gegenbauer.deriv()(XXd) * dM_cross
    dX = np.dot(X_system.T, dXXd)
    dZ = (dX - X * np.dot(X, dX)) / norm(Z)
    return dZ


def cholesky_of_gegenbauered_gram(gegenbauer_polynomial, x_matrix):
    XtX = x_matrix @ x_matrix.T
    return np.linalg.cholesky(gegenbauer_polynomial(XtX))


def normalize(vec: np.ndarray):
    assert len(vec.shape) == 1
    return vec / norm(vec)


def norm(vec: np.ndarray):
    assert len(vec.shape) == 1
    return np.sqrt(np.sum(np.square(vec)))


if __name__ == "__main__":
    from multiprocessing import Pool

    def calc_degrees(dimension: int, max_harmonics: int):
        harmonics = 0
        degree = 1
        while harmonics < max_harmonics:
            harmonics += num_harmonics(dimension, degree)
            degree += 1
        degree -= 1
        return degree

    def regenerate_cache(dimension: int):
        max_degrees = calc_degrees(dimension, max_harmonics=1000)
        FundamentalSystemCache(dimension).regenerate_and_save_cache(max_degrees)

    DIMENSIONS = range(3, 20)
    with Pool(6) as p:
        p.map(regenerate_cache, DIMENSIONS)
