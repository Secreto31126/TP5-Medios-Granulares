package org.sims.sand;

import java.util.*;

import org.sims.interfaces.*;
import org.sims.models.*;
import org.sims.neighbours.*;

public record SandEngine(SandSimulation simulation, CIM cim, Integrator<Particle, SandForce.Data> integrator)
        implements Engine<SandStep> {
    public static SandEngine build(final SandSimulation simulation) {
        final var cim = SandInitialization.cim();
        final var integrator = simulation.integrator(new SandForce.Data(cim, simulation.box()));
        return new SandEngine(simulation, cim, integrator);
    }

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
                final var hasNext = current < simulation.steps();
                if (!hasNext) {
                    cim.close();
                }

                return hasNext;
            }

            @Override
            public SandStep next() {
                time += simulation.dt();

                particles = integrator.step(particles, new SandForce.Data(cim, walls));
                particles = simulation.teleport().apply(particles, portals, cim, integrator);

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
