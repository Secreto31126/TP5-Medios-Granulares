package org.sims.sand;

import java.util.*;
import java.util.stream.*;

import org.sims.interfaces.*;
import org.sims.models.*;
import org.sims.neighbours.*;

record SandForce() implements Force<Particle, SandForce.Data> {
    public static final double Kn = 250;
    public static final double GAMMA = 1;
    public static final double MU = 0.7;

    @Override
    public Map<Particle, Vector2> apply(final Collection<Particle> particles, final Data data) {
        final var neighbours = data.cim().evaluate(particles);

        final var forces = new HashMap<Particle, Vector2>(particles.size() + 1, 1.0f);
        for (final var p : particles) {
            final var interactives = Stream
                    .concat(
                            neighbours.get(p).stream(),
                            data.walls().stream().filter(w -> w.overlap(p) > 0))
                    .toList();

            forces.merge(p, Forces.gravity(p), Vector2::add);
            forces.merge(p, Forces.overlap(p, interactives, Kn, GAMMA, MU), Vector2::add);
        }

        return forces;
    }

    public record Data(CIM cim, Collection<? extends Interactive<Particle>> walls) {
    }
}
