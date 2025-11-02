from typing import Callable

import time

import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from tqdm import tqdm

import frames
import resources
from streaming import SequentialStreamingExecutor as Executor

from classes.wall import Wall
from classes.particle import Particle

abar = None
def main(plot: bool = True, save: bool = True):
    global abar

    with open(resources.path("setup.txt")) as f:
        _steps, _aperture, _omega = [*map(float, f.readline().strip().split())]
        _steps = int(_steps)

    executor = Executor(frames.next_all, range(frames.count()))

    fig, ax = plt.subplots() # pyright: ignore[reportUnknownMemberType]
    ax.set_aspect('equal', adjustable="box")

    lines: list[Line2D] = []
    for wall in frames.next_walls(0)[1]:
        line, = ax.plot([wall.start.x, wall.end.x], [wall.start.y, wall.end.y], color="black") # pyright: ignore[reportUnknownMemberType]
        lines.append(line)

    circles: list[Circle] = []
    for p in frames.next(0)[1]:
        c = Circle(p.position.tuple(), radius=p.radius, color="blue")

        ax.add_patch(c)
        circles.append(c)

    def update(entities: tuple[list[Particle], list[Wall]]):
        global abar

        if abar is not None and abar.n % abar.total == 0:
            abar.reset()

        for i, particle in enumerate(entities[0]):
            circles[i].center = particle.position.tuple()

        for i, wall in enumerate(entities[1]):
            lines[i].set_data([wall.start.x, wall.end.x], [wall.start.y, wall.end.y])

        if abar is not None:
            abar.update()

        return circles + lines

    ani = FuncAnimation( # pyright: ignore[reportUnusedVariable]
        fig,
        update,
        frames=executor.stream(),
        save_count=frames.count(),
        interval=5,
        blit=True,
        repeat=True
    )

    if plot:
        abar = tqdm(total=frames.count())
        plt.show() # pyright: ignore[reportUnknownMemberType]
        abar.close()

    if save:
        print("Saving animation...")

        filename = resources.path(f"{int(time.time())}.mp4")
        with tqdm(total=frames.count()) as sbar:
            callback: Callable[[int, int], bool | None] = lambda _i, _n: sbar.update()
            ani.save(filename, writer='ffmpeg', fps=20, dpi=300, progress_callback=callback)

        print(f"Animation saved at {filename}")

if __name__ == "__main__":
    main("-s" not in sys.argv, "-p" not in sys.argv)
