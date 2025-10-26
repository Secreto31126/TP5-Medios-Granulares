package org.sims.models;

import java.util.*;

import org.sims.interfaces.Interactive;

public record Portal(Orientation orientation, double constant) implements Interactive<Particle> {
    @Override
    public double overlap(final Particle c) {
        final var cpos = orientation.perpendicular(c.position());
        final var dist = Math.abs(constant - cpos);
        return Math.max(c.radius() - dist, 0);
    }

    @Override
    public Vector2 position(final Particle i) {
        return orientation.normal().mult(orientation.perpendicular(i.position()));
    }

    @Override
    public Vector2 velocity(final Particle i) {
        return orientation.normal().mult(orientation.perpendicular(i.velocity()));
    }

    @Override
    public Vector2 normal(final Particle i) {
        return orientation.normal();
    }

    @Override
    public long id() {
        return 0;
    }

    @Override
    public String name() {
        return "A";
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "%.14f", constant);
    }
}
