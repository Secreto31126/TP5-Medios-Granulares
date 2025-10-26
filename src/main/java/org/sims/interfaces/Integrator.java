package org.sims.interfaces;

import java.util.*;

import org.sims.models.Particle;

/**
 * Integrators are the mathematical implementations of the
 * time evolution algorithms, such as Verlet, or Beeman.
 *
 * As the responsible of moving the entities, it must know
 * the entities type and the memory they need to save,
 * hence the existance of a nested Constructor interface.
 *
 * Constructor requires the user to provide a static class
 * that creates the integrator instance and initializes
 * the entities' memories as deemed necesary by the
 * algorithm.
 *
 * @param <E> the type of entities the integrator works with.}
 * @param <D> Additional data the force might need.
 */
public interface Integrator<E, D> extends Named {
    /**
     * Initialize the integrator based on the particles
     *
     * @param particles the particles being simulated
     * @param data      additional data the force might need
     */
    void initialize(final Collection<Particle> particles, final D data);

    /**
     * Reset a single entity's data
     *
     * @apiNote Portals don't preserve velocity or acceleration,
     *          unlike the game Portal
     *
     * @param entity The entity to reset
     */
    void reset(final E entity);

    /**
     * Advance the simulation by one time step
     *
     * @implNote The integrator MUST NOT alter the collection
     *           NOR the entities in it directly.
     *
     * @param entities the entities to move
     * @param data     additional data the force might need
     * @return the moved entities
     */
    List<E> step(final Collection<E> entities, final D data);

    /**
     * A constructor for integrators
     *
     * @param <E> the type of entities the integrator works with.
     * @param <D> Additional data the integrator might need.
     */
    public interface Constructor<E, D> {
        /**
         * Create an integrator instance
         *
         * @param entities the entities to integrate
         * @param dt       the time step
         * @param force    the force calculator
         * @return the integrator instance
         */
        Integrator<E, D> get(final double dt, final Force<E, ? extends D> force);
    }
}
