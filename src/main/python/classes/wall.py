from dataclasses import dataclass

from classes.vector import Vector

@dataclass(frozen=True, init=False)
class Wall:
    start: Vector
    end: Vector

    def __init__(self, *args: float):
        object.__setattr__(self, 'start', Vector(args[0], args[1]))
        object.__setattr__(self, 'end', Vector(args[2], args[3]))
