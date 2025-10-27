package org.sims.interfaces;

import java.io.*;
import java.util.*;

/**
 * A step in a simulation.
 *
 * Stores the state of the simulation in the step i.
 */
public interface Step {
    /**
     * The step index
     *
     * @return the step index
     */
    long i();

    /**
     * Save the step to a writer
     *
     * @param writer the writers to save to
     * @throws IOException if an I/O error occurs
     */
    void saveTo(final List<Writer> writer) throws IOException;

    /**
     * Log events that occurred during this step
     *
     * @apiNote The method is invoked for every step after saveTo,
     *          and the files must be appended to.
     *
     * @param appenders the appendable writers to log to
     * @throws IOException if an I/O error occurs
     */
    default void log(final List<Writer> appenders) throws IOException {
    }

    /**
     * Whether this step has logs to write
     *
     * @return true if there are logs to write
     */
    default boolean hasLogs() {
        return false;
    }
}
