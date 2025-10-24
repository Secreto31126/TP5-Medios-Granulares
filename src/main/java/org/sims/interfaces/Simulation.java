package org.sims.interfaces;

import java.io.*;
import java.util.*;

/**
 * Simulations defines the setup and parameters for a simulation run.
 *
 * @apiNote As the responsible of storing the simulation state,
 * it must know the steps type, the entities type and the
 * ForceCalculator implementation, so it's strongly encouraged
 * to add a static build() method that creates all the data,
 * and a record Force() implementing ForceCalculator<E>.
 *
 * @param <E> the type of the entities in the simulation
 * @param <S> the type of steps the simulation produces
 */
public interface Simulation<E, S extends Step> {
    /**
     * The number of steps in the simulation
     *
     * @return The number of steps
     */
    long steps();

    /**
     * The entities in the simulation
     *
     * @return The entities
     */
    List<E> entities();

    /**
     * The integrator used in the simulation
     *
     * @return The integrator
     */
    Integrator<E> integrator();

    /**
     * Save the simulation setup to a writer
     * 
     * @param writer the writer to save to
     * @throws IOException if an I/O error occurs
     */
    void saveTo(final Writer writer) throws IOException;
}
