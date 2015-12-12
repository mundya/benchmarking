import argparse
import nengo
from nengo import spa
import nengo_spinnaker
import numpy as np
from six import iteritems
import time


def make_model(n_dimensions, seed):
    """Create a model with the given parameters."""
    with spa.SPA(seed=seed) as model:
        # Create the inputs
        model.a = spa.State(dimensions=n_dimensions)
        model.b = spa.State(dimensions=n_dimensions)

        # Create the output
        model.c = spa.State(dimensions=n_dimensions)

        # Create the convolution
        model.cortical = spa.Cortical(spa.Actions("c = a * b"), synapse=0.01)

        # Add some input
        model.input = spa.Input(a='A', b='B')

    return model


def run_experiment(model, spinnaker):
    """Run the experiment and return the results in a dictionary."""
    # Add probes for the output of the network
    with model:
        p_c = nengo.Probe(model.c.output, synapse=0.01)

    # Create the simulator
    if not spinnaker:
        sim = nengo.Simulator(model)
    else:
        # Configure all the Nodes as function of time Nodes
        nengo_spinnaker.add_spinnaker_params(model.config)

        for n in model.all_nodes:
            if n.output is not None:
                model.config[n].function_of_time = True

        sim = nengo_spinnaker.Simulator(model)

        # Get the current number of dropped packets
        dropped = sum(
            sim.controller.get_router_diagnostics(x, y).dropped_multicast
            for (x, y) in sim.controller.get_machine()
        )

    # Run the simulation
    sim.run(0.5)

    # Get the vocabulary
    vocab = model.get_output_vocab("c")
    AB = vocab.parse("A * B").v

    # Prepare the results for later analysis
    results = dict()
    results["output"] = np.dot(AB, sim.data[p_c].T).T
    results["times"] = sim.trange()

    # Tidy up SpiNNaker and get the count of dropped packets
    if spinnaker:
        sim.close()

        # Count the number of packets dropped during the simulation
        results["dropped_multicast"] = sum(
            sim.controller.get_router_diagnostics(x, y).dropped_multicast
            for (x, y) in sim.controller.get_machine()
        ) - dropped

    return results


def run_all_experiments(n_dimensions, spinnaker=False, runs_per_scale=30):
    """Run a large number of experiments."""
    # Initialise the results as empty lists
    data = {"n_dimensions": list(),
            "times": None,
            "output": list()}
    if spinnaker:
        data["dropped_multicast"] = list()

    # Repeatedly build and run the experiments
    for n_dims in n_dimensions:
        for i in range(runs_per_scale):
            model = make_model(n_dims, i)
            results = run_experiment(model, spinnaker)

            # Combine the results with the already stored results
            data["n_dimensions"].append(n_dims)
            for k, v in iteritems(results):
                if k == "times":
                    if data["times"] is None:
                        data["times"] = v
                else:
                    data[k].append(v)

    # Store all the data in Numpy arrays and write to file
    final_data = dict()
    final_data["n_dimensions"] = np.array(data.pop("n_dimensions"),
                                          dtype=np.int)

    for k, v in iteritems(data):
        final_data[k] = np.array(v)

    np.savez_compressed(
        "convolution_{}_{}.npz".format("nengo" if not spinnaker else "spinnaker",
                                       int(time.time())),
        **final_data
    )


if __name__ == "__main__":
    # Get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("n_dimensions", nargs="+", type=int)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--spinnaker", action="store_true")
    args = parser.parse_args()

    # Run the experiment
    run_all_experiments(args.n_dimensions,
                        args.spinnaker,
                        args.runs)
