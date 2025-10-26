package org.sims.integrals;

import java.util.*;
import java.util.concurrent.*;

import org.sims.interfaces.*;
import org.sims.models.*;

public record Beeman<D>(double dt, double dt2, Force<Particle, D> force, Map<Particle, Vector2> memory)
        implements Integrator<Particle, D> {
    public Beeman(final double dt, final Force<Particle, D> force, Map<Particle, Vector2> memory) {
        this(dt, dt * dt, force, memory);
    }

    public Beeman(final double dt, final Force<Particle, D> force) {
        this(dt, force, new ConcurrentHashMap<>());
    }

    @Override
    public void initialize(final Collection<Particle> particles, final D data) {
        memory.putAll(force.apply(Beeman.prev(particles, this.dt), data));
    }

    @Override
    public void reset(Particle entity) {
        memory.put(entity, Vector2.ZERO);    
    }

    @Override
    public List<Particle> step(final Collection<Particle> particles, D data) {
        final var prediction = particles.parallelStream().map(p -> {
            final var predicted_pos = p.position()
                    .add(p.velocity().mult(dt))
                    .add(p.acceleration().mult((2.0 / 3.0) * dt2))
                    .subtract(memory.get(p).mult((1.0 / 6.0) * dt2));

            final var predicted_vel = p.velocity()
                    .add(p.acceleration().mult((3.0 / 2.0) * dt))
                    .subtract(memory.get(p).mult((1.0 / 2.0) * dt));

            return new Particle(p, predicted_pos, predicted_vel, p.acceleration());
        }).toList();

        final var accs = force.apply(prediction, data);

        return particles.parallelStream().map(p -> {
            // Dupped code :/
            final var future_pos = p.position()
                    .add(p.velocity().mult(dt))
                    .add(p.acceleration().mult((2.0 / 3.0) * dt2))
                    .subtract(memory.get(p).mult((1.0 / 6.0) * dt2));

            final var future_acc = accs.get(p);

            final var future_vel = p.velocity()
                    .add(future_acc.mult((1.0 / 3.0) * dt))
                    .add(p.acceleration().mult((5.0 / 6.0) * dt))
                    .subtract(memory.get(p).mult((1.0 / 6.0) * dt));

            memory.put(p, p.acceleration());
            return new Particle(p, future_pos, future_vel, future_acc);
        }).toList();
    }

    @Override
    public String name() {
        return "Beeman";
    }

    /**
     * Get the particles at time t - dt
     *
     * @param particles The particles
     * @return The particles at time t - dt
     */
    private static List<Particle> prev(final Collection<Particle> particles, final double dt) {
        return particles.parallelStream().map(p -> Beeman.before(p, dt)).toList();
    }

    /**
     * Get the the particle at time t - dt
     *
     * @param p  The particle
     * @param dt The time step
     * @return The particle at time t - dt
     */
    private static Particle before(final Particle p, final double dt) {
        final var pos = p.position().subtract(p.velocity().mult(dt)).subtract(p.acceleration().mult(dt * dt / 2));
        final var vel = p.velocity().subtract(p.acceleration().mult(dt));
        return new Particle(p, pos, vel, p.acceleration());
    }
}
