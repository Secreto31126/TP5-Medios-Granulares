from typing import Callable, TypeVar

from functools import cache
import os

import pandas as pd

from classes.particle import Particle
from classes.wall import Wall
import resources

T = TypeVar('T', bound='Particle | Wall')
def next(f: int, dir: str = 'particles', kclass: Callable[..., T] = Particle) -> tuple[int, list[T]]:
    """
    Reads the input file for a given frame.

    Args:
        f (int): The frame number to read.
        dir (str, optional): The directory where the frame files are located. Defaults to 'particles'.
        kclass (Callable[..., T], optional): The class to instantiate for each entry. Defaults to Particle.
    Returns:
        tuple[int, list[T]]: A tuple containing the frame number and a list of instantiated objects.
    """
    file_path = resources.path(dir, f"{f}.txt")
    df = pd.read_csv(file_path, header=None, delimiter=' ').astype("float") # type: ignore[reportUnknownMemberType]
    return f, [kclass(*d) for _, d in df.iterrows()]

def next_particles(f: int):
    """
    Reads the input file from the 'particles' folder for a given frame and returns particles.
    """
    return next(f, 'particles', Particle)

def next_walls(f: int):
    """
    Reads the input file from the 'walls' folder for a given frame and returns walls.
    """
    return next(f, 'walls', Wall)

def next_all(f: int):
    """
    Reads the input files for a given frame and returns both particles and walls.
    """
    return f, (next_particles(f)[1], next_walls(f)[1])

@cache
def count(dir: str = 'particles'):
    """
    Returns the number of animations steps.
    """
    return len(os.listdir(resources.path(dir)))
