package org.sims.neighbours;

import java.util.*;
import java.util.concurrent.*;

import org.sims.models.*;

public record CIM(Mapping mapping, double Rc, ExecutorService executor) implements AutoCloseable {
    /**
     * Default constructor with 8 threads
     *
     * @param width  The width of the simulation box
     * @param height The height of the simulation box
     * @param Rc     The interaction radius
     */
    public CIM(final double width, final double height, final double Rc) {
        this(new Mapping(width, height, Rc), Rc, Executors.newFixedThreadPool(6));
    }

    /**
     * Reset the mapping by clearing all quadrants
     */
    public void reset() {
        mapping.matrix().forEach(List::clear);
    }

    /**
     * Add a particle to the mapping
     *
     * @param p Particle to add
     */
    public void add(final Particle p) {
        mapping.add(p);
    }

    /**
     * Evaluates the interaction between particles in a simulation box.
     *
     * @apiNote Mappings are reset before evaluation.
     * @apiNote Particles are NOT neighbours to themselves.
     *
     * @param particles List of particles to evaluate
     * @return A map where each key is a particle and the value is a list of
     *         particles that interact with it.
     */
    public Map<Particle, List<Particle>> evaluate(final Collection<Particle> particles) {
        final var tasks = new ArrayList<Callable<Object>>(particles.size());
        final var result = new ConcurrentHashMap<Particle, List<Particle>>();

        this.reset();
        for (final var p : particles) {
            final var coordinates = mapping.add(p);
            final var neighbours = new LinkedList<Particle>();

            result.put(p, neighbours);
            tasks.add(Executors.callable(new Task(p, result, Rc, mapping, coordinates)));
        }

        try {
            executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            throw new RuntimeException("Evaluation interrupted", e);
        }

        return result;
    }

    /**
     * Finds any interaction of a single particle in the mapping.
     *
     * This method is useful for checking if a newly added particle
     * interacts with any existing particles in the mapping.
     *
     * @apiNote Mappings are NOT populated before evaluation.
     * @apiNote Particle is NOT neighbour to itself.
     *
     * @param p The particle to evaluate
     * @return The particle's first found neighbour
     */
    public Particle evaluate(final Particle p) {
        final var pos = mapping.getCoordinates(p);

        for (var coord : mapping.longList(pos)) {
            final var quadrant = mapping.matrix().get((int) coord.x(), (int) coord.y());
            for (final var other : quadrant) {
                if (p.id() != other.id() && CIM.distance(p, other) < Rc) {
                    return other;
                }
            }
        }

        return null;
    }

    @Override
    public void close() {
        executor.close();
    }

    private static record Task(Particle p, Map<Particle, List<Particle>> result, double Rc,
            Mapping mapping, List<Vector2> coordinates) implements Runnable {
        @Override
        public void run() {
            for (final var coords : coordinates) {
                final var quadrant = mapping.matrix().get((int) coords.x(), (int) coords.y());

                for (final var other : quadrant) {
                    if (p.id() < other.id() && CIM.distance(p, other) < Rc) {
                        result.compute(p, (_, list) -> {
                            list.add(other);
                            return list;
                        });

                        result.compute(other, (p, list) -> {
                            list.add(p);
                            return list;
                        });
                    }
                }
            }
        }
    }

    /**
     * Calculate the distance between two particles considering their radius
     *
     * @param a The first particle
     * @param b The second particle
     * @return The distance between the surfaces of the two particles
     */
    private static double distance(final Particle a, final Particle b) {
        return a.position().subtract(b.position()).norm() - (a.radius() + b.radius());
    }
}
