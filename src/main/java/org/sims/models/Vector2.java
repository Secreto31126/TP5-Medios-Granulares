package org.sims.models;

import java.util.Locale;

public record Vector2(double x, double y) {
    public static final Vector2 NONE_NONE = new Vector2(-1.0, -1.0);
    public static final Vector2 NONE_ZERO = new Vector2(-1.0, 0.0);
    public static final Vector2 NONE_ONE = new Vector2(-1.0, 1.0);

    public static final Vector2 ZERO_NONE = new Vector2(0.0, -1.0);
    public static final Vector2 ZERO_ZERO = new Vector2(0.0, 0.0);
    public static final Vector2 ZERO_ONE = new Vector2(0.0, 1.0);

    public static final Vector2 ONE_NONE = new Vector2(1.0, -1.0);
    public static final Vector2 ONE_ZERO = new Vector2(1.0, 0.0);
    public static final Vector2 ONE_ONE = new Vector2(1.0, 1.0);

    public static final Vector2 ZERO = ZERO_ZERO;

    public Vector2 neg() {
        return new Vector2(-x, -y);
    }

    public Vector2 add(final Vector2 v) {
        return new Vector2(x + v.x, y + v.y);
    }

    public Vector2 subtract(final Vector2 v) {
        return new Vector2(x - v.x, y - v.y);
    }

    public Vector2 mult(final double scalar) {
        return new Vector2(x * scalar, y * scalar);
    }

    public static Vector2 mult(final double scalar, final Vector2 v) {
        return v.mult(scalar);
    }

    public Vector2 div(final double scalar) {
        return new Vector2(x / scalar, y / scalar);
    }

    public static Vector2 div(final double scalar, final Vector2 v) {
        return v.div(scalar);
    }

    public double dot(final Vector2 v) {
        return x * v.x + y * v.y;
    }

    public double norm2() {
        return this.dot(this);
    }

    public double norm() {
        return Math.sqrt(this.norm2());
    }

    public Vector2 normalize() {
        return this.div(this.norm());
    }

    public Vector2 hadamard(final Vector2 v) {
        return new Vector2(x * v.x, y * v.y);
    }

    /**
     * @apiNote Rotate the vector 90 degrees counter-clockwise
     */
    public Vector2 rotate() {
        return new Vector2(-y, x);
    }

    public Vector2 rotate(final double angle) {
        final var sin = Math.sin(angle);
        final var cos = Math.cos(angle);
        return new Vector2(cos - sin, sin + cos).hadamard(this);
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "%+.14f %+.14f", x, y);
    }
}
