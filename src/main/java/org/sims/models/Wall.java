package org.sims.models;

import java.util.function.*;

import org.sims.interfaces.Interactive;

// I hate the velocity field
public record Wall(long id, Orientation orientation, Vector2 a, Vector2 b,
        double velocity, Function<Double, Double[]> movement)
        implements Interactive<Particle> {
    private static long SERIAL = 0L;

    /**
     * Create a new wall with movement and initial velocity 0.
     *
     * @param orientation The orientation of the wall
     * @param a           The start point
     * @param b           The end point
     * @param movement    The movement function of the wall (time -> [pos, vel])
     */
    public Wall(final Orientation orientation, final Vector2 a, final Vector2 b,
            final Function<Double, Double[]> movement) {
        this(SERIAL++, orientation, a, b, 0, movement);
    }

    /**
     * Create a new static wall.
     *
     * @param orientation The orientation of the wall
     * @param a           The start point
     * @param b           The end point
     */
    public Wall(final Orientation orientation, final Vector2 a, final Vector2 b) {
        this(SERIAL++, orientation, a, b, 0, null);
    }

    /**
     * Update the wall position according to its movement function
     *
     * @apiNote The walls can only move in the direction parallel to their
     *          orientation.
     *
     * @param dt time delta
     * @return the updated wall, self if static
     */
    public Wall update(final double dt) {
        if (this.movement == null) {
            return this;
        }

        final var moved = this.movement.apply(dt);
        final var a = this.orientation.move_constant(this.a, moved[0]);
        final var b = this.orientation.move_constant(this.b, moved[0]);

        return new Wall(this.id, this.orientation, a, b, moved[1], this.movement);
    }

    @Override
    public double overlap(final Particle p) {
        if (!between(this.orientation.parallel(p.position()), p.radius(),
                this.orientation.variable(a), this.orientation.variable(b))) {
            return 0.0;
        }

        final var cwall = this.orientation.constant(a);
        final var cpos = this.orientation.perpendicular(p.position());

        final var dist = Math.abs(cwall - cpos);
        return Math.max(p.radius() - dist, 0);
    }

    @Override
    public Vector2 position(final Particle p) {
        return this.orientation.tangencial()
                .mult(this.orientation.constant(a) - this.orientation.perpendicular(p.position()) - p.radius());
    }

    @Override
    public Vector2 velocity(final Particle p) {
        return this.orientation.tangencial()
                .mult(this.velocity - this.orientation.perpendicular(p.velocity()));
    }

    @Override
    public Vector2 normal(final Particle p) {
        final var sign = Math.signum(this.orientation.constant(a) - this.orientation.perpendicular(p.position()));
        return this.orientation.normal().mult(sign);
    }

    /**
     * Checks if a value is between two other values, inclusive,
     * regardless of order
     *
     * @param val   value to check
     * @param range Small range of tolerance
     * @param a     first bound
     * @param b     second bound
     * @return true if val is between a and b
     */
    private static final boolean between(final double val, final double range, final double a, final double b) {
        return Math.min(a, b) - range <= val && val <= Math.max(a, b) + range;
    }

    @Override
    public String name() {
        return "W";
    }

    @Override
    public String toString() {
        return "%s %s".formatted(a, b);
    }

    @Override
    public final boolean equals(Object obj) {
        if (this == obj)
            return true;

        if (!(obj instanceof Wall other))
            return false;

        return this.id == other.id;
    }

    @Override
    public final int hashCode() {
        return Long.hashCode(id);
    }
}
