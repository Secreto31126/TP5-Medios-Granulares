package org.sims.sand;

import java.util.*;

import org.sims.interfaces.Engine;
import org.sims.models.*;

public record SandEngine(SandSimulation simulation) implements Engine<SandStep> {
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

            @Override
            public boolean hasNext() {
                return current < simulation.steps();
            }

            @Override
            public SandStep next() {
                particles = simulation.integrator().step(particles);

                time += SandSimulation.DT;
                walls = walls.stream().map(w -> w.update(time)).toList();

                return new SandStep(++current, particles, walls);
            }
        };
    }

    @Override
    public void close() throws Exception {
    }
}
