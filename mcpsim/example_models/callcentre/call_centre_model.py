
"""
Enhanced Call Centre Simulation Model with Nurse Callbacks

A discrete event simulation model of a call centre using SimPy.
Extended to include nurse callbacks for 40% of patients.

Author: Enhanced from Tom Monks' original model
"""

import numpy as np
import pandas as pd
import simpy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

# =============================================================================
# CONSTANTS AND DEFAULT VALUES
# =============================================================================

# Default resources
N_OPERATORS = 13
N_NURSES = 10

# Default mean inter-arrival time (exp)
MEAN_IAT = 60 / 100

# Default service time parameters (triangular)
CALL_LOW = 5.0
CALL_MODE = 7.0
CALL_HIGH = 10.0

# Nurse callback parameters
CALLBACK_PROBABILITY = 0.4  # 40% of patients need nurse callback
NURSE_CONSULT_LOW = 10.0    # Uniform distribution 10-20 minutes
NURSE_CONSULT_HIGH = 20.0

# Sampling settings
N_STREAMS = 4  # Increased for additional distributions
DEFAULT_RND_SET = 0

# Boolean switch to display simulation results as the model runs
TRACE = False

# Run variables
RESULTS_COLLECTION_PERIOD = 1000

# =============================================================================
# DISTRIBUTION CLASSES
# =============================================================================

class Triangular:
    """
    Convenience class for the triangular distribution.
    Packages up distribution parameters, seed and random generator.
    """
    def __init__(self, low, mode, high, random_seed=None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high
        self.mode = mode

    def sample(self, size=None):
        return self.rand.triangular(self.low, self.mode, self.high, size=size)

class Exponential:
    """
    Convenience class for the exponential distribution.
    Packages up distribution parameters, seed and random generator.
    """
    def __init__(self, mean, random_seed=None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.mean = mean

    def sample(self, size=None):
        return self.rand.exponential(self.mean, size=size)

class Uniform:
    """
    Convenience class for the uniform distribution.
    Packages up distribution parameters, seed and random generator.
    """
    def __init__(self, low, high, random_seed=None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high

    def sample(self, size=None):
        return self.rand.uniform(self.low, self.high, size=size)

class Bernoulli:
    """
    Convenience class for the Bernoulli distribution.
    Used for callback decision (40% probability).
    """
    def __init__(self, p, random_seed=None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.p = p

    def sample(self, size=None):
        return self.rand.binomial(1, self.p, size=size)

# =============================================================================
# EXPERIMENT CLASS
# =============================================================================

class Experiment:
    """
    Enhanced experiment class with nurse callback functionality.
    """
    def __init__(
        self,
        random_number_set=DEFAULT_RND_SET,
        n_operators=N_OPERATORS,
        n_nurses=N_NURSES,
        mean_iat=MEAN_IAT,
        call_low=CALL_LOW,
        call_mode=CALL_MODE,
        call_high=CALL_HIGH,
        callback_prob=CALLBACK_PROBABILITY,
        nurse_consult_low=NURSE_CONSULT_LOW,
        nurse_consult_high=NURSE_CONSULT_HIGH,
        n_streams=N_STREAMS,
    ):
        # sampling
        self.random_number_set = random_number_set
        self.n_streams = n_streams

        # store parameters for the run of the model
        self.n_operators = n_operators
        self.n_nurses = n_nurses
        self.mean_iat = mean_iat
        self.call_low = call_low
        self.call_mode = call_mode
        self.call_high = call_high
        self.callback_prob = callback_prob
        self.nurse_consult_low = nurse_consult_low
        self.nurse_consult_high = nurse_consult_high

        # resources: initialized after Environment is created
        self.operators = None
        self.nurses = None

        # initialise results and sampling
        self.init_results_variables()
        self.init_sampling()

    def set_random_no_set(self, random_number_set):
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        """Create the distributions used by the model"""
        # produce n non-overlapping streams
        seed_sequence = np.random.SeedSequence(self.random_number_set)
        self.seeds = seed_sequence.spawn(self.n_streams)

        # create distributions
        self.arrival_dist = Exponential(self.mean_iat, random_seed=self.seeds[0])
        self.call_dist = Triangular(
            self.call_low, self.call_mode, self.call_high, random_seed=self.seeds[1]
        )
        self.callback_dist = Bernoulli(self.callback_prob, random_seed=self.seeds[2])
        self.nurse_dist = Uniform(
            self.nurse_consult_low, self.nurse_consult_high, random_seed=self.seeds[3]
        )

    def init_results_variables(self):
        """Initialize all experiment variables used in results collection"""
        self.results = {}

        # Original results
        self.results["waiting_times"] = []
        self.results["total_call_duration"] = 0.0

        # New nurse results
        self.results["nurse_waiting_times"] = []
        self.results["total_nurse_duration"] = 0.0
        self.results["callbacks_requested"] = 0
        self.results["total_patients"] = 0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def trace(msg):
    """Turning printing of events on and off."""
    if TRACE:
        print(msg)

# =============================================================================
# MODEL LOGIC
# =============================================================================

def nurse_service(identifier, env, args):
    """
    Simulates the nurse callback process
    1. request and wait for a nurse
    2. nurse consultation (uniform)
    3. exit system
    """
    # record the time that patient entered nurse queue
    start_wait = env.now

    # request a nurse
    with args.nurses.request() as req:
        yield req

        # record the waiting time for nurse callback
        waiting_time = env.now - start_wait
        args.results["nurse_waiting_times"].append(waiting_time)

        trace(f"nurse started callback for patient {identifier} at {env.now:.3f}")

        # nurse consultation time
        consult_duration = args.nurse_dist.sample()
        yield env.timeout(consult_duration)

        # update total nurse duration
        args.results["total_nurse_duration"] += consult_duration

        trace(f"nurse callback for patient {identifier} ended at {env.now:.3f}; "
              f"waiting time was {waiting_time:.3f}")

def service(identifier, env, args):
    """
    Enhanced service process with nurse callback
    1. request and wait for a call operator
    2. phone triage (triangular)
    3. determine if callback needed (40% probability)
    4. if needed, initiate nurse callback process
    5. exit system
    """
    # record the time that call entered the queue
    start_wait = env.now

    # request an operator
    with args.operators.request() as req:
        yield req

        # record the waiting time for call to be answered
        waiting_time = env.now - start_wait
        args.results["waiting_times"].append(waiting_time)

        trace(f"operator answered call {identifier} at {env.now:.3f}")

        # call duration
        call_duration = args.call_dist.sample()
        yield env.timeout(call_duration)

        # update the total call duration
        args.results["total_call_duration"] += call_duration

        trace(f"call {identifier} ended {env.now:.3f}; "
              f"waiting time was {waiting_time:.3f}")

    # After operator call, determine if nurse callback is needed
    args.results["total_patients"] += 1
    callback_needed = args.callback_dist.sample()

    if callback_needed:
        args.results["callbacks_requested"] += 1
        trace(f"patient {identifier} requires nurse callback")

        # Start nurse callback process
        env.process(nurse_service(identifier, env, args))

def arrivals_generator(env, args):
    """Generate patient arrivals with exponential inter-arrival times"""
    for caller_count in itertools.count(start=1):
        # sample inter-arrival time
        inter_arrival_time = args.arrival_dist.sample()
        yield env.timeout(inter_arrival_time)

        trace(f"call arrives at: {env.now:.3f}")

        # start service process
        env.process(service(caller_count, env, args))

# =============================================================================
# EXPERIMENT EXECUTION FUNCTIONS
# =============================================================================

def single_run(experiment, rep=0, rc_period=RESULTS_COLLECTION_PERIOD):
    """Perform a single run of the enhanced model"""
    run_results = {}

    # reset all result collection variables
    experiment.init_results_variables()

    # set random number set
    experiment.set_random_no_set(rep)

    # create environment
    env = simpy.Environment()

    # create resources
    experiment.operators = simpy.Resource(env, capacity=experiment.n_operators)
    experiment.nurses = simpy.Resource(env, capacity=experiment.n_nurses)

    # start arrivals process
    env.process(arrivals_generator(env, experiment))

    # run simulation
    env.run(until=rc_period)

    # calculate results
    run_results["01_mean_waiting_time"] = np.mean(experiment.results["waiting_times"])
    run_results["02_operator_util"] = (
        experiment.results["total_call_duration"] / 
        (rc_period * experiment.n_operators)
    ) * 100.0

    # Calculate nurse results
    if experiment.results["nurse_waiting_times"]:
        run_results["03_mean_nurse_waiting_time"] = np.mean(experiment.results["nurse_waiting_times"])
    else:
        run_results["03_mean_nurse_waiting_time"] = 0.0

    run_results["04_nurse_util"] = (
        experiment.results["total_nurse_duration"] / 
        (rc_period * experiment.n_nurses)
    ) * 100.0

    # Additional metrics
    run_results["05_callback_rate"] = (
        experiment.results["callbacks_requested"] / 
        experiment.results["total_patients"]
    ) * 100.0 if experiment.results["total_patients"] > 0 else 0.0

    return run_results

def multiple_replications(experiment, rc_period=RESULTS_COLLECTION_PERIOD, n_reps=5):
    """Perform multiple replications of the enhanced model"""
    results = [single_run(experiment, rep, rc_period) for rep in range(n_reps)]
    df_results = pd.DataFrame(results)
    df_results.index = np.arange(1, len(df_results) + 1)
    df_results.index.name = "rep"
    return df_results

def set_trace(trace_on=True):
    """Turn tracing on/off globally"""
    global TRACE
    TRACE = trace_on


def run_simulation_from_dict(params: dict):
    """
    Wrapper to run the simulation using params supplied in a dictionary.
    Only the parameters necessary for the Experiment's constructor are passed;
    defaults are used for those not provided.
    """
    # Extract constructor signature of the Experiment class
    signature = inspect.signature(Experiment.__init__)
    
    # Build kwargs dict: match keys in 'params' to Experiment.__init__ params
    experiment_kwargs = {
        key: params.get(key, param.default)
        for key, param in signature.parameters.items()
        if key != 'self'  # exclude 'self' from constructor
    }

    # Create Experiment instance with dynamic/default parameters
    exp = Experiment(**experiment_kwargs)

    # Separate non-constructor parameters
    run_length = params.get("run_length", RESULTS_COLLECTION_PERIOD)
    rep_seed = params.get("random_seed", DEFAULT_RND_SET)

    # Run simulation
    return single_run(exp, rep=rep_seed, rc_period=run_length)

