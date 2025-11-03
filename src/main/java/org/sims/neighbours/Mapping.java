package org.sims.neighbours;

import java.util.*;
import java.util.concurrent.*;

import org.sims.models.*;

record Mapping(Matrix<Queue<Particle>> matrix, double cells_w, double cells_h) {
    private static final Vector2[] LONG_LIST_OFFSETS = {
            Vector2.NONE_NONE,
            Vector2.NONE_ZERO,
            Vector2.NONE_ONE,
            Vector2.ZERO_NONE,
            Vector2.ZERO_ZERO,
            Vector2.ZERO_ONE,
            Vector2.ONE_NONE,
            Vector2.ONE_ZERO,
            Vector2.ONE_ONE
    };

    private static final Vector2[] SHORT_LIST_OFFSETS = {
            Vector2.NONE_NONE,
            Vector2.NONE_ZERO,
            Vector2.NONE_ONE,
            Vector2.ZERO_NONE,
            Vector2.ZERO_ZERO
    };

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

        var x_trunc = (int) coord.x();
        if (coord.x() < 0) {
            x_trunc = 0;
        } else if (this.matrix.rows() <= coord.x()) {
            x_trunc = this.matrix.rows() - 1;
        }

        var y_trunc = (int) coord.y();
        if (coord.y() < 0) {
            y_trunc = 0;
        } else if (this.matrix.cols() <= coord.y()) {
            y_trunc = this.matrix.cols() - 1;
        }

        this.matrix.get(x_trunc, y_trunc).add(p);
        return this.longList(coord);
    }

    Vector2 getCoordinates(final Particle p) {
        final var i = (int) (p.position().x() / this.cells_w);
        final var j = (int) (p.position().y() / this.cells_h);
        return new Vector2(i, j);
    }

    @Deprecated
    List<Vector2> shortList(final Vector2 coord) {
        final var result = new ArrayList<Vector2>(SHORT_LIST_OFFSETS.length);
        for (var offset : SHORT_LIST_OFFSETS) {
            final var c = coord.add(offset);
            if (this.inbound(c)) {
                result.add(c);
            }
        }

        return result;
    }

    List<Vector2> longList(final Vector2 coord) {
        final var result = new ArrayList<Vector2>(LONG_LIST_OFFSETS.length);
        for (var offset : LONG_LIST_OFFSETS) {
            final var c = coord.add(offset);
            if (this.inbound(c)) {
                result.add(c);
            }
        }

        return result;
    }

    private boolean inbound(final Vector2 coord) {
        return inbound(coord.x(), this.matrix.rows()) && inbound(coord.y(), this.matrix.cols());
    }

    private boolean inbound(final double i, final int max) {
        return 0 <= i && i < max;
    }
}
