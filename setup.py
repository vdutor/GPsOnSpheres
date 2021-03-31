from setuptools import setup

requirements = [
    "gpflow>=2.1",
    "numpy",
    "scipy",
    "tensorflow-probability>=0.12.0",
    "tensorflow>=2.4.0",
]


name = 'gspheres'

setup(
    name="GPs on Spheres",
    version="0.01",
    author="vd309@cam.ac.uk",
    description="Gaussian processes on Spheres",
    install_requires=requirements,
    packages=[name],
)
