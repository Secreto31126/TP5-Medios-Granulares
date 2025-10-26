package org.sims;

import java.util.*;

import org.sims.sand.*;

import me.tongfei.progressbar.ProgressBar;

public class Main {
    private static final long SAVE_INTERVAL = 100L;

    public static void main(String[] args) throws Exception {
        usage(args);

        final var steps = Long.parseLong(args[0]);
        final var aperture = Double.parseDouble(args[1]);
        final var omega = Double.parseDouble(args[2]);

        final var simulation = new SandSimulation(steps, 200, aperture, omega);

        final var pbs = new ProgressBar("Vibrating", simulation.steps());
        final var pbw = new ProgressBar("Writting", simulation.steps() / SAVE_INTERVAL + 1);

        final var onStep = new Orchestrator.SkipSteps(SAVE_INTERVAL, pbs::step);
        try (pbw; pbs; final var engine = SandEngine.build(simulation)) {
            new Orchestrator(simulation, engine, List.of("particles", "walls")).start(onStep, pbw::step);
        }
    }

    private static void usage(final String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java -jar tp5.jar <steps> <omega>");
            System.err.println("\t<steps>: The number of steps to simulate");
            System.err.println("\t<aperture>: The aperture of the box");
            System.err.println("\t<omega>: The vibration frequency");
            System.exit(1);
        }
    }
}
