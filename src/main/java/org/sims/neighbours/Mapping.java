package org.sims.neighbours;

import java.util.*;

import org.sims.models.*;

public record Mapping(Matrix<List<Particle>> matrix, double cells_w, double cells_h) {
    public Mapping(double width, double height, double Rc) {
        this(
                new Matrix<>((int) (width / Rc), (int) (height / Rc), LinkedList<Particle>::new),
                width / (int) (width / Rc),
                height / (int) (height / Rc));
    }

    /**
     * Add a particle to the mapping
     *
     * @param p Particle to add
     * @return The short list of coordinates to check for neighbours
     */
    public List<Vector2> add(final Particle p) {
        final var coord = getCoordinates(p);
        matrix.get((int) coord.x(), (int) coord.y()).add(p);
        return Mapping.shortList(coord);
    }

    Vector2 getCoordinates(final Particle p) {
        final var i = (int) (p.position().x() / cells_w);
        final var j = (int) (p.position().y() / cells_h);
        return new Vector2(i, j);
    }

    static List<Vector2> shortList(Vector2 coord) {
        return List.of(
                coord.add(Vector2.NONE_NONE),
                coord.add(Vector2.NONE_ZERO),
                coord.add(Vector2.NONE_ONE),
                coord.add(Vector2.ZERO_NONE),
                coord);
    }

    static List<Vector2> longList(Vector2 coord) {
        return List.of(
                coord.add(Vector2.NONE_NONE),
                coord.add(Vector2.NONE_ZERO),
                coord.add(Vector2.NONE_ONE),
                coord.add(Vector2.ZERO_NONE),
                coord.add(Vector2.ZERO_ZERO),
                coord.add(Vector2.ZERO_ONE),
                coord.add(Vector2.ONE_NONE),
                coord.add(Vector2.ONE_ZERO),
                coord.add(Vector2.ONE_ONE));
    }
}
