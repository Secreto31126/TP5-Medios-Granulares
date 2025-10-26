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
    public static final double MU = 0.25;

    @Override
    public Map<Particle, Vector2> apply(
            final Collection<Particle> particles,
            final Data data) {
        final var neighbours = data.cim().evaluate(particles);

        return particles.parallelStream()
                .collect(Collectors.toMap(Function.identity(), p -> {
                    final var walls = data.walls().stream().filter(w -> w.overlap(p) > 0);

                    final var interactives = Stream.concat(walls, neighbours.get(p).stream()).toList();

                    final var gravity = Forces.gravity(p);
                    final var overlap = Forces.overlap(p, interactives, Kn, GAMMA, MU);

                    return gravity.add(overlap);
                }));
    }

    public record Data(CIM cim, Collection<? extends Interactive<Particle>> walls) {
    }
}
