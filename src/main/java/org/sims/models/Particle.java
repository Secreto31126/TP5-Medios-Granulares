package org.sims.models;

import java.util.Locale;

import org.sims.interfaces.*;

/**
 * A particle in 2D space with position, velocity, acceleration, and radius.
 *
 * Each particle has a unique ID, auto-assigned on creation.
 *
 * The events counter tracks the number of collision events involving this
 * particle.
 */
public record Particle(long id, Vector2 position, Vector2 velocity, Vector2 acceleration, double radius, double mass)
        implements Interactive<Particle> {
    private static long SERIAL = 0L;

    /**
     * Create a new particle with a unique ID.
     *
     * @param position     The initial position
     * @param velocity     The initial velocity
     * @param acceleration The initial acceleration
     * @param radius       The radius of the particle
     * @param mass         The mass of the particle
     */
    public Particle(final Vector2 position, final Vector2 velocity, final Vector2 acceleration, final double radius, final double mass) {
        this(SERIAL++, position, velocity, acceleration, radius, mass);
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
        this(p.id, position, velocity, acceleration, p.radius, p.mass);
    }

    /**
     * Generates a ghost particle at a given position and radius.
     *
     * @apiNote The ghost particle has id zero (not counted towards SERIAL),
     *          zero velocity, zero acceleration and mass zero.
     *
     * @param position The position of the ghost particle
     * @param radius   The radius of the ghost particle
     */
    public Particle(final Vector2 position, final double radius) {
        this(0, position, Vector2.ZERO, Vector2.ZERO, radius, 0);
    }

    /**
     * Generates a ghost particle at a given position.
     *
     * @apiNote The ghost particle has id zero (not counted towards SERIAL),
     *          zero velocity, zero acceleration, radius zero, and mass zero.
     *
     * @param position The position of the ghost particle
     */
    public Particle(final Vector2 position) {
        this(position, 0.0);
    }

    @Override
    public double overlap(final Particle c) {
        if (this.equals(c)) {
            return 0.0;
        }

        return (this.radius + c.radius) - this.position.subtract(c.position).norm();
    }

    @Override
    public Vector2 position(final Particle i) {
        return this.position.subtract(i.position);
    }

    @Override
    public Vector2 velocity(final Particle i) {
        return this.velocity.subtract(i.velocity);
    }

    @Override
    public Vector2 normal(final Particle i) {
        return this.position(i).normalize();
    }

    @Override
    public String name() {
        return "P";
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "%s %s %+.14f", position, velocity, radius);
    }

    @Override
    public final boolean equals(final Object obj) {
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
