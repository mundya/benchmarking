import argparse
import logging
import nengo
from nengo import spa
import nengo_spinnaker
import numpy as np
from six import iteritems
import time


def make_model(n_dimensions, n_symbols, time_per_symbol, seed):
    """Create a model with the given parameters."""
    with spa.SPA(seed=seed) as model:
        # Inputs to the memory
        model.key = spa.State(dimensions=n_dimensions)
        model.val = spa.State(dimensions=n_dimensions)

        # The memory itself
        model.mem = spa.State(dimensions=n_dimensions, feedback=1.0)

        # Cue and output from the memory
        model.cue = spa.State(dimensions=n_dimensions)
        model.out = spa.State(dimensions=n_dimensions)

        # Add the connections between the elements
        model.cortical = spa.Cortical(
            spa.Actions("mem = key * val",
                        "out = mem * ~cue")
        )

        # Create the inputs to the model
        def make_input(char, delay=0.0):
            """Get a new input function."""
            def f(t):
                index = int((t - delay) / time_per_symbol)
                if 0 <= index < n_symbols:
                    return "{}{}".format(char, index)
                else:
                    return "0"

            return f

        model.input = spa.Input(
            val=make_input("V"),
            key=make_input("K"),
            cue=make_input("K", delay=time_per_symbol * n_symbols)
        )

    return model


def run_experiment(model, spinnaker, n_symbols, time_per_symbol):
    """Run the experiment and return the results in a dictionary."""
    # Add a probe for the output of the network
    with model:
        p_out = nengo.Probe(model.out.output, synapse=0.03)

    # Create the simulator
    if not spinnaker:
        sim = nengo.Simulator(model)
    else:
        # Configure all the Nodes as function of time Nodes
        nengo_spinnaker.add_spinnaker_params(model.config)

        for n in model.all_nodes:
            if n.output is not None:
                model.config[n].function_of_time = True

        sim = nengo_spinnaker.Simulator(model, use_spalloc=True)

        # Get the current number of dropped packets
        dropped = sum(
            sim.controller.get_router_diagnostics(x, y).dropped_multicast
            for (x, y) in sim.controller.get_machine()
        )

    # Run the simulation
    sim.run(2*(n_symbols + 1)*time_per_symbol)

    # Prepare the results for later analysis
    results = dict()

    vocab = model.get_output_vocab("out")
    mem_out = np.array(sim.data[p_out])

    # Get an ordered representation of the keys
    m_vocab = np.zeros((n_symbols*2, vocab.dimensions))
    for n in range(n_symbols):
        m_vocab[n] = vocab["K{}".format(n)].v
        m_vocab[n + n_symbols] = vocab["V{}".format(n)].v

    results["output"] = np.dot(m_vocab, mem_out.T).T
    results["times"] = sim.trange()

    # Tidy up SpiNNaker and get the count of dropped packets
    if spinnaker:
        # Count the number of packets dropped during the simulation
        results["dropped_multicast"] = sum(
            sim.controller.get_router_diagnostics(x, y).dropped_multicast
            for (x, y) in sim.controller.get_machine()
        ) - dropped

        sim.close()

    return results


def run_all_experiments(n_dimensions, spinnaker=False, n_symbols=4,
                        time_per_symbol=0.2, runs_per_scale=30):
    """Run a large number of experiments."""
    # Initialise the results as empty lists
    data = {"n_dimensions": list(),
            "times": None,
            "output": list(),
            "n_symbols": n_symbols,
            "time_per_symbol": time_per_symbol}
    if spinnaker:
        data["dropped_multicast"] = list()

    # Repeatedly build and run the experiments
    for n_dims in n_dimensions:
        for i in range(runs_per_scale):
            model = make_model(n_dims, n_symbols, time_per_symbol, i)
            results = run_experiment(model, spinnaker,
                                     n_symbols, time_per_symbol)

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
    for k in ["n_dimensions", "n_symbols"]:
        final_data[k] = np.array(data.pop(k), dtype=np.int)

    for k, v in iteritems(data):
        final_data[k] = np.array(v)

    np.savez_compressed(
        "recall_{}_{}.npz".format("nengo" if not spinnaker else "spinnaker",
                                  int(time.time())),
        **final_data
    )


if __name__ == "__main__":
    # Get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("n_dimensions", nargs="+", type=int)
    parser.add_argument("--n_symbols", type=int, default=4)
    parser.add_argument("--time_per_symbol", type=float, default=0.2)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--spinnaker", action="store_true")
    parser.add_argument("-v", "--verbosity", action="count")
    args = parser.parse_args()

    if args.verbosity is not None:
        if args.verbosity == 1:
            logging.basicConfig(level=logging.INFO)
        if args.verbosity >= 2:
            logging.basicConfig(level=logging.DEBUG)

    # Run the experiment
    run_all_experiments(args.n_dimensions,
                        args.spinnaker,
                        args.n_symbols,
                        args.time_per_symbol,
                        args.runs)
