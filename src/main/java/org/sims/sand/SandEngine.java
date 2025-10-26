package org.sims.sand;

import java.util.*;

import org.sims.interfaces.*;
import org.sims.models.*;
import org.sims.neighbours.*;

public record SandEngine(SandSimulation simulation, CIM cim) implements Engine<SandStep> {
    @Override
    public SandStep initial() {
        return new SandStep(0, simulation.entities(), simulation.box());
    }

    @Override
    public Iterator<SandStep> iterator() {
        return new Iterator<>() {
            private long current = 0;
            private double time = 0.0;

            private List<Particle> particles = simulation.entities();
            private List<Wall> walls = simulation.box();
            private List<Portal> portals = simulation.portals();

            @Override
            public boolean hasNext() {
                return current < simulation.steps();
            }

            @Override
            public SandStep next() {
                time += simulation.dt();

                particles = simulation.integrator().step(particles, new SandForce.Data(cim, walls));
                particles = simulation.teleport().apply(particles, portals, cim);

                walls = walls.stream().map(w -> w.update(time)).toList();

                return new SandStep(++current, particles, walls);
            }
        };
    }

    @Override
    public void close() throws Exception {
        cim.close();
    }
}
