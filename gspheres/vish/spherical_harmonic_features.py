from gpflow.inducing_variables import InducingVariables

from gspheres.spherical_harmonics import SphericalHarmonics


class SphericalHarmonicFeatures(InducingVariables):
    """Wraps SphericalHarmonics."""
    def __init__(self, dimension: int, degrees: int):
        self.dimension = dimension
        self.max_degree = degrees
        self.spherical_harmonics = SphericalHarmonics(dimension, degrees)

    def __len__(self):
        """Number of inducing variables"""
        return len(self.spherical_harmonics)
