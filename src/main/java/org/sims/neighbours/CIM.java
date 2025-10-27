package org.sims.neighbours;

import java.util.*;
import java.util.function.*;
import java.util.stream.*;

import org.sims.models.*;

public record CIM(Mapping mapping, double Rc, BiFunction<Particle, Particle, Boolean> interacting) {
    /**
     * Default constructor with 6 threads
     *
     * @param width       The width of the simulation box
     * @param height      The height of the simulation box
     * @param Rc          The interaction search radius
     * @param interacting The interaction function between two particles
     */
    public CIM(final double width, final double height, final double Rc,
            BiFunction<Particle, Particle, Boolean> interacting) {
        this(new Mapping(width, height, Rc), Rc, interacting);
    }

    /**
     * Add a particle to the mapping
     *
     * @param p Particle to add
     */
    public void add(final Particle p) {
        this.mapping.add(p);
    }

    /**
     * Reset the mapping
     */
    public void reset() {
        this.mapping.clear();
    }

    /**
     * Calculates all the neighbour particles in a simulation box.
     *
     * @apiNote Mappings are reset before evaluation.
     *
     * @implNote The returned map is concurrent.
     *
     * @param particles List of particles to evaluate
     * @return A map where each key is a particle and the value is a list of
     *         particles that interact with it.
     */
    public Map<Particle, List<Particle>> evaluate(final Collection<Particle> particles) {
        this.reset();
        final var coords = particles.parallelStream()
                .collect(Collectors.toMap(Function.identity(), this.mapping::add));
        return coords.entrySet().parallelStream()
                .collect(Collectors.toConcurrentMap(Map.Entry::getKey, this::interacting));
    }

    /**
     * Finds any neighbour of a single particle in the mapping.
     *
     * This method is useful for checking if a newly added particle
     * interacts with any existing particles in the mapping.
     * The algorithm is slightly faster as it exits on the first match.
     *
     * @apiNote Mappings are NOT populated before evaluation.
     *
     * @param p The particle to evaluate
     * @return Whether the particle has any interacting neighbours
     */
    public boolean evaluate(final Particle p) {
        return this.mapping.longList(this.mapping.getCoordinates(p)).stream()
                .flatMap(c -> this.mapping.matrix().get(c).stream())
                .anyMatch(o -> this.interacting(p, o));
    }

    /**
     * Finds all interacting particles for a given entry.
     *
     * @param e The entry mapping a particle to its short list of cells
     * @return A list of interacting particles
     */
    private List<Particle> interacting(Map.Entry<Particle, List<Vector2>> e) {
        return this.interacting(e.getKey(), e.getValue());
    }

    /**
     * Finds all interacting particles for a given particle and its short list of
     * cells.
     *
     * @param p The particle to evaluate
     * @param q The short list of cells to check
     * @return A list of interacting particles
     */
    private List<Particle> interacting(Particle p, List<Vector2> q) {
        return q.parallelStream()
                // Get all particles in the listed cells
                .flatMap(c -> this.mapping.matrix().get(c).stream())
                // Preserve only interacting particles
                .filter(o -> this.interacting(p, o))
                .toList();
    }

    /**
     * Checks if two particles are interacting,
     * based on the provided interaction function.
     *
     * @param a The first particle
     * @param b The second particle
     * @return Whether they are interacting
     */
    private boolean interacting(final Particle a, final Particle b) {
        return this.interacting().apply(a, b);
    }
}
