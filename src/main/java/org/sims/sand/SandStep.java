package org.sims.sand;

import java.io.*;
import java.util.*;

import org.sims.interfaces.*;
import org.sims.models.*;

/**
 * A step in the simulation
 *
 * @param i         the step index
 * @param t         the time at this step
 * @param particles the particles in the step
 * @param walls     the walls in the step
 * @param exited    the particles that have exited in this step
 */
public record SandStep(long i, double t, Collection<Particle> particles, Collection<Wall> walls,
        Collection<Particle> exited)
        implements Step {
    @Override
    public void saveTo(final List<Writer> writers) throws IOException {
        for (final var p : particles) {
            writers.get(0).write("%s\n".formatted(p));
        }

        for (final var w : walls) {
            writers.get(1).write("%s\n".formatted(w));
        }
    }

    @Override
    public void log(final List<Writer> writers) throws IOException {
        if (!exited.isEmpty()) {
            writers.get(0).append(String.format(Locale.ROOT, "%+.14f %d\n", t, exited.size()));
        }

    }
}
