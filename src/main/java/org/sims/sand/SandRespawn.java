package org.sims.sand;

import java.util.*;

import org.sims.models.*;
import org.sims.neighbours.*;

public interface SandRespawn {
    /**
     * Apply the respawning of particles that have exited through portals
     *
     * @param particles The current particles
     * @param portals   The portals in the simulation
     * @param cim       The CIM for superposition checks
     * @return The updated list of particles after respawning
     */
    List<Particle> apply(Collection<Particle> particles, Collection<Portal> portals, CIM cim);
}
