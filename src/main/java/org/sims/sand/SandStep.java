package org.sims.sand;

import java.io.*;
import java.util.*;

import org.sims.interfaces.*;
import org.sims.models.*;

/**
 * A step in the simulation
 *
 * @param i the step index
 */
public record SandStep(long i, Collection<Particle> particles, Collection<Wall> walls) implements Step {
    @Override
    public void saveTo(final List<Writer> writers) throws IOException {
        for (final var p : particles) {
            writers.get(0).write("%s\n".formatted(p));
        }

        for (final var w : walls) {
            writers.get(1).write("%s\n".formatted(w));
        }
    }
}
