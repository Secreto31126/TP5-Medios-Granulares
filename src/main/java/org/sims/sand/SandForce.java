package org.sims.sand;

import java.util.*;
import java.util.function.*;
import java.util.stream.*;

import org.sims.interfaces.*;
import org.sims.models.*;

record SandForce() implements Force<Particle> {
    @Override
    public Map<Particle, Vector2> apply(final Collection<Particle> particles) {
        return particles.parallelStream().collect(Collectors.toMap(Function.identity(), p -> {
            return Forces.gravity(p, particles);
        }));
    }
}
