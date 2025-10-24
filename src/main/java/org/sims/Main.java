package org.sims;

import java.util.List;

import org.sims.sand.SandEngine;
import org.sims.sand.SandSimulation;

import me.tongfei.progressbar.ProgressBar;

public class Main {
    private static final long SAVE_INTERVAL = 10L;

    public static void main(String[] args) throws Exception {
        usage(args);

        final var steps = Long.parseLong(args[0]);
        final var omega = Double.parseDouble(args[1]);

        final var simulation = SandSimulation.build(steps, 200, omega);

        final var pbs = new ProgressBar("Vibrating", simulation.steps());
        final var pbw = new ProgressBar("Writting", simulation.steps() / SAVE_INTERVAL + 1);

        final var onStep = new Orchestrator.SkipSteps(SAVE_INTERVAL, pbs::step);
        try (pbw; pbs; final var engine = new SandEngine(simulation)) {
            new Orchestrator(simulation, engine, List.of("particles", "walls")).start(onStep, pbw::step);
        }
    }

    private static void usage(final String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java -jar tp5.jar <steps> <omega>");
            System.err.println("\t<steps>: The number of steps to simulate");
            System.err.println("\t<omega>: The vibration frequency");
            System.exit(1);
        }
    }
}
