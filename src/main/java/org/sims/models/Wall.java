package org.sims.models;

import java.util.*;
import java.util.function.*;

import org.sims.interfaces.Collideable;

public record Wall(long id, Wall.Orientation orientation, Vector2 a, Vector2 b, Function<Double, Double> movement)
        implements Collideable<Particle> {
    private static long SERIAL = 0L;

    /**
     * Create a new wall with movement.
     *
     * @param orientation The orientation of the wall
     * @param a           The start point
     * @param b           The end point
     * @param movement    The movement function of the wall
     */
    public Wall(Orientation orientation, Vector2 a, Vector2 b, Function<Double, Double> movement) {
        this(SERIAL++, orientation, a, b, movement);
    }

    /**
     * Create a new static wall.
     *
     * @param orientation The orientation of the wall
     * @param a           The start point
     * @param b           The end point
     */
    public Wall(Orientation orientation, Vector2 a, Vector2 b) {
        this(SERIAL++, orientation, a, b, null);
    }

    public static enum Orientation {
        VER(Vector2.ONE_ZERO, Vector2.ZERO_ONE),
        HOR(Vector2.ZERO_ONE, Vector2.ONE_ZERO);

        private final Vector2 constant;
        private final Vector2 variable;

        Orientation(final Vector2 constant, final Vector2 variable) {
            this.constant = constant;
            this.variable = variable;
        }

        /**
         * Get the constant coordinate of a vector according to the orientation type
         *
         * @param v vector to get coordinate from
         * @return constant value
         */
        public double constant(Vector2 v) {
            return v.dot(constant);
        }

        /**
         * Get the variable coordinate of a vector according to the orientation type
         *
         * @param v vector to get coordinate from
         * @return variable value
         */
        public double variable(Vector2 v) {
            return v.dot(variable);
        }

        /**
         * Collide a velocity vector according to the wall orientation
         *
         * @param v vector to collide
         * @return collided vector
         */
        public Vector2 collide(Vector2 v) {
            return v.hadamard(constant).neg().add(v.hadamard(variable));
        }

        /**
         * Move a vector according to the wall orientation
         *
         * @param v        vector to move
         * @param constant new constant displacement
         * @return moved vector
         */
        public Vector2 move_constant(Vector2 v, double constant) {
            return this.constant.mult(constant).add(v.hadamard(variable));
        }
    }

    /**
     * Update the wall position according to its movement function
     *
     * @apiNote The walls can only move in the direction parallel to their
     * orientation.
     *
     * @param dt time delta
     * @return the updated wall
     */
    public Wall update(double dt) {
        if (movement == null) {
            return this;
        }

        final var pos = movement.apply(dt);
        final var a = orientation.move_constant(this.a, pos);
        final var b = orientation.move_constant(this.b, pos);

        return new Wall(orientation, a, b, movement);
    }

    /**
     * @implNote This might look like magic, but it's just
     *           taking advantage of the fact that the logic is the
     *           same for vertical and horizontal walls, just
     *           swapping x and y
     */
    @Override
    public double collisionTime(final Particle p) {
        // Get the constant coordinate of the wall (x for vertical, y for horizontal)
        final var wall = orientation.constant(a);
        // Get the constant coordinates of the particle's position and velocity
        final var cvel = orientation.constant(p.velocity());
        final var cpos = orientation.constant(p.position());

        // Check if the particle is moving towards the wall
        final var bottom_left = cpos < wall && cvel > 0;
        final var top_right = wall < cpos && cvel < 0;
        // These variables read as "the particle comes from a or b"

        if (!bottom_left && !top_right) {
            return Double.POSITIVE_INFINITY;
        }

        // If the particle comes from bottom/left, the radius is subtracted,
        // otherwise added
        final var r_effect = (bottom_left ? -1 : +1) * p.radius();
        final var time = (wall + r_effect - cpos) / cvel;

        // Never happens, top_left and bottom_right ensures the division is positive
        // if (time < 0) return Double.POSITIVE_INFINITY;

        // Get the future position of the particle at collision time
        // TODO: See if this needs to use integrator prediction
        final var future = p.position().add(p.velocity().mult(time)).add(p.acceleration().mult(time * time / 2));

        // If the particle in the future is not between the wall's length
        if (!between(orientation.variable(future), p.radius(), orientation.variable(a), orientation.variable(b))) {
            return Double.POSITIVE_INFINITY;
        }

        // Finally, peace
        return time;
    }

    @Override
    public List<Particle> collide(final Particle p) {
        p.events().incrementAndGet();
        return List.of(p.velocity(orientation.collide(p.velocity())));
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
    private static final boolean between(double val, double range, double a, double b) {
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
