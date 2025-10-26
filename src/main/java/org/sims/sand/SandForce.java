package org.sims.sand;

import java.util.*;
import java.util.function.*;
import java.util.stream.*;

import org.sims.interfaces.*;
import org.sims.models.*;
import org.sims.neighbours.*;

record SandForce() implements Force<Particle, SandForce.Data> {
    public static final double Kn = 250;
    public static final double GAMMA = 1;
    public static final double MU = 0.7;

    @Override
    public Map<Particle, Vector2> apply(
            final Collection<Particle> particles,
            final Data data) {
        // final var neighbours = data.cim().evaluate(particles);

        return particles.parallelStream()
                .collect(Collectors.toMap(Function.identity(), p -> {
                    final var interactives = Stream
                            .concat(particles.stream(), data.walls().stream())
                            .filter(o -> o.overlap(p) > 0)
                            .toList();

                    final var gravity = Forces.gravity(p);
                    final var overlap = Forces.overlap(p, interactives, Kn, GAMMA, MU);

                    return gravity.add(overlap);
                }));
    }

    public record Data(CIM cim, Collection<? extends Interactive<Particle>> walls) {
    }
}
