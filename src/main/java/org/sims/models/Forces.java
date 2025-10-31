package org.sims.models;

import java.util.*;

import org.sims.interfaces.Interactive;

public abstract class Forces {
    private static final Vector2 EARTH_GRAVITY_VECTOR = new Vector2(0, -10);

    /**
     * Get the earth gravity force
     *
     * @param p The particles
     * @return The gravitational force exerted on the particle
     */
    public static Vector2 gravity(final Particle p) {
        return EARTH_GRAVITY_VECTOR.mult(p.mass());
    }

    /**
     * Get the overlap force of all the overlaping particles
     *
     * @apiNote The interactives collection must not contain non-overlaping entities
     *
     * @param p     The particle
     * @param inter The collection of interactives
     * @param Kn    Spring constant
     * @param gamma Damping constant
     * @param mu    Friction coefficient
     * @return
     */
    public static Vector2 overlap(final Particle p, final Collection<Interactive<Particle>> inter,
            final double Kn,
            final double gamma,
            final double mu) {
        return inter.stream()
                .map(i -> {
                    final var normal = i.normal(p);
                    final var tangent = normal.rotate();

                    final var velocity = i.velocity(p);

                    final var Fn = normal.mult(-Kn * i.overlap(p) - velocity.dot(normal) * gamma);
                    final var Ft = tangent.mult(-mu * Fn.norm() * Math.signum(velocity.dot(tangent)));

                    return Fn.add(Ft);
                })
                .reduce(Vector2.ZERO, Vector2::add);
    }
}
