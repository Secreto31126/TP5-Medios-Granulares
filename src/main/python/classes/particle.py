from dataclasses import dataclass

from classes.vector import Vector

@dataclass(frozen=True, init=False)
class Particle:
    position: Vector
    velocity: Vector
    radius: float
    mass: float

    def __init__(self, *args: float):
        object.__setattr__(self, 'position', Vector(args[0], args[1]))
        object.__setattr__(self, 'velocity', Vector(args[2], args[3]))
        object.__setattr__(self, 'radius', args[4])
        object.__setattr__(self, 'mass', 0.001)
