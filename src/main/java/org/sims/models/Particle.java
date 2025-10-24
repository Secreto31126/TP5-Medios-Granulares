package org.sims.models;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

import org.sims.interfaces.*;

/**
 * A particle in 2D space with position, velocity, acceleration, and radius.
 *
 * Each particle has a unique ID, auto-assigned on creation.
 *
 * The events counter tracks the number of collision events involving this
 * particle.
 */
public record Particle(long id, Vector2 position, Vector2 velocity, Vector2 acceleration, double radius,
        AtomicLong events)
        implements Collideable<Particle> {
    private static long SERIAL = 0L;

    /**
     * Create a new particle with a unique ID.
     *
     * @param position     The initial position
     * @param velocity     The initial velocity
     * @param acceleration The initial acceleration
     * @param radius       The radius of the particle
     */
    public Particle(final Vector2 position, final Vector2 velocity, final Vector2 acceleration, final double radius) {
        this(SERIAL++, position, velocity, acceleration, radius, new AtomicLong(0L));
    }

    /**
     * Copy constructor with new position, velocity and memory.
     *
     * @apiNote The new particle will preserve the ID, radius
     *          and events of the original.
     *
     * @param p            The particle to copy
     * @param position     The new position
     * @param velocity     The new velocity
     * @param acceleration The new acceleration
     */
    public Particle(final Particle p, final Vector2 position, final Vector2 velocity, final Vector2 acceleration) {
        this(p.id, position, velocity, acceleration, p.radius, p.events);
    }

    /**
     * Generates a ghost particle at a given position and radius.
     *
     * @apiNote The ghost particle has id zero (not counted towards SERIAL),
     *          zero velocity, zero acceleration, radius zero, and events is null.
     *
     * @param position The position of the ghost particle
     * @param radius   The radius of the ghost particle
     */
    public Particle(final Vector2 position, double radius) {
        this(0, position, Vector2.ZERO, Vector2.ZERO, radius, null);
    }

    /**
     * Generates a ghost particle at a given position.
     *
     * @apiNote The ghost particle has id zero (not counted towards SERIAL),
     *          zero velocity, zero acceleration, radius zero, and events is null.
     *
     * @param position The position of the ghost particle
     */
    public Particle(final Vector2 position) {
        this(position, 0.0);
    }

    /**
     * @apiNote Returns Double.POSITIVE_INFINITY if there is no collision.
     */
    @Override
    public double collisionTime(final Particle p) {
        if (p == this) {
            return Double.POSITIVE_INFINITY;
        }

        final var rvel = p.velocity.subtract(this.velocity);
        final var rpos = p.position.subtract(this.position);

        final var vel_pos = rvel.dot(rpos);

        if (vel_pos >= -1e-14) {
            return Double.POSITIVE_INFINITY;
        }

        final var vel_vel = rvel.dot(rvel);

        if (-1e-14 < vel_vel && vel_vel < 1e-14) {
            return Double.POSITIVE_INFINITY;
        }

        final var pos_pos = rpos.dot(rpos);
        final var sigma = this.radius + p.radius;

        final var d = vel_pos * vel_pos - vel_vel * (pos_pos - sigma * sigma);

        if (d < 1e-14) {
            return Double.POSITIVE_INFINITY;
        }

        final var t = -(vel_pos + Math.sqrt(d)) / vel_vel;

        if (t < 1e-14) {
            return Double.POSITIVE_INFINITY;
        }

        return t;
    }

    @Override
    public List<Particle> collide(final Particle p) {
        this.events().incrementAndGet();
        p.events().incrementAndGet();

        final var rvel = p.velocity.subtract(this.velocity);
        final var rpos = p.position.subtract(this.position);

        final var vel_pos = rvel.dot(rpos);
        final var dist = this.radius + p.radius;

        final var impulse = (2 * vel_pos) / (2 * dist);
        final var j = rpos.mult(impulse).div(dist);

        return List.of(this.velocity(this.velocity.add(j)),
                p.velocity(p.velocity().subtract(j)));
    }

    /**
     * Create a new particle with the same parameters as this, but with
     * a new position.
     *
     * @param newPosition The new position
     * @return The new particle
     */
    public Particle position(final Vector2 newPosition) {
        return new Particle(this, newPosition, this.velocity, this.acceleration);
    }

    /**
     * Create a new particle with the same parameters as this, but with
     * a new velocity.
     *
     * @param newVelocity The new velocity
     * @return The new particle
     */
    public Particle velocity(final Vector2 newVelocity) {
        return new Particle(this, this.position, newVelocity, this.acceleration);
    }

    /**
     * Create a new particle with the same parameters as this, but with
     * a new acceleration.
     *
     * @param newAcceleration The new acceleration
     * @return The new particle
     */
    public Particle acceleration(final Vector2 newAcceleration) {
        return new Particle(this, this.position, this.velocity, newAcceleration);
    }

    @Override
    public String name() {
        return "P";
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "%s %s %.14f", position, velocity, radius);
    }

    @Override
    public final boolean equals(Object obj) {
        if (this == obj)
            return true;

        if (!(obj instanceof Particle o))
            return false;

        return this.id == o.id;
    }

    @Override
    public final int hashCode() {
        return Long.hashCode(this.id);
    }
}
