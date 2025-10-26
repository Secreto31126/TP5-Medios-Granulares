package org.sims.integrals;

import java.util.*;
import java.util.stream.*;
import java.util.function.*;

import org.sims.interfaces.*;
import org.sims.models.*;

public record Gear5<D>(double dt, Force<Particle, D> force, Map<Particle, List<Vector2>> memory)
        implements Integrator<Particle, D> {
    private static final List<Double> COEF5 = List.of(
            /* a0 = */ 3.0 / 16.0,
            /* a1 = */ 251.0 / 360.0,
            /* a2 = */ 1.0,
            /* a3 = */ 11.0 / 18.0,
            /* a4 = */ 1.0 / 6.0,
            /* a5 = */ 1.0 / 60.0);

    public Gear5(final Collection<Particle> particles, final double dt, final Force<Particle, D> force, final D data) {
        this(dt, force, particles.parallelStream()
                .collect(Collectors.toConcurrentMap(Function.identity(), Gear5::initials)));
    }

    @Override
    public void reset(final Particle entity) {
        memory.put(entity, initials(entity));
    }

    @Override
    public List<Particle> step(final Collection<Particle> particles, final D data) {
        final var dt1 = dt;
        final var dt2 = dt1 * dt;
        final var dt3 = dt2 * dt;
        final var dt4 = dt3 * dt;
        final var dt5 = dt4 * dt;

        final var moved = particles.parallelStream().map(p -> {
            final var pos = p.position()
                    .add(p.velocity().mult(dt1))
                    .add(this.r(p, 2).mult(dt2 / 2.0))
                    .add(this.r(p, 3).mult(dt3 / 6.0))
                    .add(this.r(p, 4).mult(dt4 / 24.0))
                    .add(this.r(p, 5).mult(dt5 / 120.0));

            final var vel = p.velocity()
                    .add(this.r(p, 2).mult(dt1))
                    .add(this.r(p, 3).mult(dt2 / 2.0))
                    .add(this.r(p, 4).mult(dt3 / 6.0))
                    .add(this.r(p, 5).mult(dt4 / 24.0));

            final var r2 = this.r(p, 2)
                    .add(this.r(p, 3).mult(dt1))
                    .add(this.r(p, 4).mult(dt2 / 2.0))
                    .add(this.r(p, 5).mult(dt3 / 6.0));

            final var r3 = this.r(p, 3)
                    .add(this.r(p, 4).mult(dt1))
                    .add(this.r(p, 5).mult(dt2 / 2.0));

            final var r4 = this.r(p, 4)
                    .add(this.r(p, 5).mult(dt1));

            final var r5 = this.r(p, 5);

            memory.put(p, List.of(r2, r3, r4, r5));
            return new Particle(p, pos, vel, p.acceleration());
        }).toList();

        final var forces = force.apply(moved, data);

        return moved.parallelStream().map(p -> {
            final var acc = forces.get(p).div(p.mass());

            final var d = acc.subtract(this.r(p, 2));

            final var pos = p.position().add(d.mult(COEF5.get(0) * dt2 / 2.0));
            final var vel = p.velocity().add(d.mult(COEF5.get(1) * dt / 2.0));

            final var r2 = this.r(p, 2).add(d.mult(COEF5.get(2)));
            final var r3 = this.r(p, 3).add(d.mult(COEF5.get(3) * 3.0 / dt));
            final var r4 = this.r(p, 4).add(d.mult(COEF5.get(4) * 12.0 / dt2));
            final var r5 = this.r(p, 5).add(d.mult(COEF5.get(5) * 60.0 / dt3));

            memory.put(p, List.of(r2, r3, r4, r5));
            return new Particle(p, pos, vel, acc);
        }).toList();
    }

    @Override
    public String name() {
        return "Gear5";
    }

    private Vector2 r(final Particle p, final int n) {
        return memory.get(p).get(n - 2);
    }

    private static List<Vector2> initials(final Particle p) {
        return List.<Vector2>of(p.acceleration(), p.velocity(), Vector2.ZERO, Vector2.ZERO);
    }
}
