from dataclasses import dataclass

@dataclass(frozen=True)
class Vector:
    x: float
    y: float

    def tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __neg__(self):
        return Vector(-self.x, self.y)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return self + (-other)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vector':
        return self * scalar

    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(self.x / scalar, self.y / scalar)

    def dot(self, other: 'Vector') -> float:
        return self.x * other.x + self.y * other.y

    def norm2(self) -> float:
        return self.dot(self)

    def norm(self) -> float:
        return self.norm2() ** 0.5

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y
