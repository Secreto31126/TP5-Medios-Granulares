package org.sims.models;

import java.util.*;
import java.util.function.*;

public abstract class Forces {
    public static final double G = 1;
    public static final double H = 0.05;

    /**
     * Newtonian gravity with softening
     *
     * @apiNote Assumes G = 1, m1 = m2 = 1
     *
     * @param p1 The first particle
     * @param p2 The second particle
     * @return The force exerted by p2 on p1
     */
    public static Vector2 gravity(final Particle p1, final Particle p2) {
        final var rij = p1.position().subtract(p2.position());
        final var r2 = rij.norm2();
        final var factor = -Math.pow(r2 + H * H, 1.5);
        return rij.div(factor);
    }

    /**
     * Compute the gravitational force exerted by
     * the universe of particles on a single particle
     *
     * @param particles The universe of particles
     * @return The total gravitational force exerted on this particle
     */
    public static Vector2 gravity(final Particle p, final Collection<Particle> particles) {
        return particles.parallelStream()
                .filter(Predicate.not(p::equals))
                .map(o -> Forces.gravity(p, o))
                .reduce(Vector2.ZERO, Vector2::add);
    }
}
