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
}
