package org.sims.neighbours;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

import org.sims.models.*;

record Mapping(Matrix<Queue<Particle>> matrix, double cells_w, double cells_h) {
    public Mapping(double width, double height, double Rc) {
        this(
                new Matrix<>((int) (width / Rc), (int) (height / Rc), ConcurrentLinkedQueue<Particle>::new),
                width / (int) (width / Rc),
                height / (int) (height / Rc));
    }

    public void clear() {
        this.matrix.set(ConcurrentLinkedQueue<Particle>::new);
    }

    /**
     * Add a particle to the mapping
     *
     * @param p Particle to add
     * @return The short list of coordinates to check for neighbours
     */
    public List<Vector2> add(final Particle p) {
        final var coord = this.getCoordinates(p);

        var x_cheese = (int) coord.x();
        if (coord.x() < 0) {
            System.err.println("Particle under bounds on X: " + p.position().x());
            x_cheese = 0;
        } else if (this.matrix.rows() <= coord.x()) {
            System.err.println("Particle over bounds on X: " + p.position().x());
            x_cheese = this.matrix.rows() - 1;
        }

        var y_cheese = (int) coord.y();
        if (coord.y() < 0) {
            System.err.println("Particle under bounds on X: " + p.position().y());
            y_cheese = 0;
        } else if (this.matrix.cols() <= coord.y()) {
            System.err.println("Particle over bounds on X: " + p.position().y());
            y_cheese = this.matrix.cols() - 1;
        }

        this.matrix.get(x_cheese, y_cheese).add(p);
        return this.longList(coord);
    }

    Vector2 getCoordinates(final Particle p) {
        final var i = (int) (p.position().x() / this.cells_w);
        final var j = (int) (p.position().y() / this.cells_h);
        return new Vector2(i, j);
    }

    @Deprecated
    List<Vector2> shortList(final Vector2 coord) {
        return filterOutOfBounds(Stream.of(
                coord.add(Vector2.NONE_NONE),
                coord.add(Vector2.NONE_ZERO),
                coord.add(Vector2.NONE_ONE),
                coord.add(Vector2.ZERO_NONE),
                coord.add(Vector2.ZERO_ZERO)));
    }

    List<Vector2> longList(final Vector2 coord) {
        return filterOutOfBounds(Stream.of(
                coord.add(Vector2.NONE_NONE),
                coord.add(Vector2.NONE_ZERO),
                coord.add(Vector2.NONE_ONE),
                coord.add(Vector2.ZERO_NONE),
                coord.add(Vector2.ZERO_ZERO),
                coord.add(Vector2.ZERO_ONE),
                coord.add(Vector2.ONE_NONE),
                coord.add(Vector2.ONE_ZERO),
                coord.add(Vector2.ONE_ONE)));
    }

    private List<Vector2> filterOutOfBounds(final Stream<Vector2> coords) {
        return coords
                .filter(c -> inbound(c.x(), this.matrix.rows()) && inbound(c.y(), this.matrix.cols()))
                .toList();
    }

    private boolean inbound(final double i, final int max) {
        return 0 <= i && i < max;
    }
}
