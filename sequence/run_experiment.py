import argparse
import logging
import nengo
from nengo import spa
import nengo_spinnaker
import numpy as np
from six import iteritems
import time


def label_net(network, prefixes=tuple()):
    for net in network.networks:
        label_net(net, prefixes + (net.label, ))

    prefix = ".".join(p or "?" for p in prefixes)
    for node in network.nodes:
        node.label = "{}.{}".format(prefix, node.label)


def make_model(n_dimensions, seed):
    """Create a model with the given parameters."""
    with spa.SPA(seed=seed) as model:
        # Create the state holding element
        model.state = spa.State(dimensions=n_dimensions,
                                feedback=1.0, feedback_synapse=0.01)

        # Create the state transitions
        actions = spa.Actions(*("dot(state, {}) --> state = {}".format(x, y) for
                                (x, y) in zip("ABCDE", "BCDEA")))
        model.bg = spa.BasalGanglia(actions=actions)
        model.thal = spa.Thalamus(model.bg)

        # Create the input for the initial state
        model.input = spa.Input(state=lambda t: 'A' if t < 0.05 else '0')

    label_net(model)

    return model


def run_experiment(model, spinnaker):
    """Run the experiment and return the results in a dictionary."""
    # Add probes for the output of the network
    with model:
        p_actions = nengo.Probe(model.thal.actions.output, synapse=0.01)
        p_utility = nengo.Probe(model.bg.input, synapse=0.01)

    # Create the simulator
    if not spinnaker:
        sim = nengo.Simulator(model)
    else:
        # Configure all the Nodes as function of time Nodes
        nengo_spinnaker.add_spinnaker_params(model.config)

        for n in model.all_nodes:
            if n.output is not None:
                model.config[n].function_of_time = True
            elif n.label[:2] == "bg" or "state.state." in n.label:
                model.config[n].optimize_out = True
            elif "thal.actions.output" in n.label:
                model.config[n].optimize_out = False

        sim = nengo_spinnaker.Simulator(model)

        # Get the current number of dropped packets
        dropped = sum(
            sim.controller.get_router_diagnostics(x, y).dropped_multicast
            for (x, y) in sim.controller.get_machine()
        )

    # Run the simulation
    sim.run(1.0)

    # Prepare the results for later analysis
    results = dict()
    results["actions"] = sim.data[p_actions]
    results["utility"] = sim.data[p_utility]
    results["times"] = sim.trange()

    # Tidy up SpiNNaker and get the count of dropped packets
    if spinnaker:
        sim.close()

        # Count the number of packets dropped during the simulation
        results["dropped_multicast"] = sum(
            sim.controller.get_router_diagnostics(x, y).dropped_multicast
            for (x, y) in sim.controller.get_machine()
        ) - dropped
        print("> Dropped {}".format(results["dropped_multicast"]))

    return results


def run_all_experiments(n_dimensions, spinnaker=False, runs_per_scale=30):
    """Run a large number of experiments."""
    # Initialise the results as empty lists
    data = {"n_dimensions": list(),
            "times": None,
            "actions": list(),
            "utility": list()}
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
        "sequence_{}_{}.npz".format("nengo" if not spinnaker else "spinnaker",
                                    int(time.time())),
        **final_data
    )


if __name__ == "__main__":
    # Get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("n_dimensions", nargs="+", type=int)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--spinnaker", action="store_true")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    args = parser.parse_args()

    if args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)

    # Run the experiment
    run_all_experiments(args.n_dimensions,
                        args.spinnaker,
                        args.runs)
