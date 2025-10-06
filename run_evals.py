import asyncio
import json
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from agent_self_reflection import build_graph, AgentState
from langchain_ollama import OllamaLLM
from mcpsim.tracing import init_tracing
import phoenix as px
from phoenix.trace import SpanEvaluations

import argparse
import itertools

# Import OpenTelemetry trace API to capture span context
from opentelemetry import trace

# Initialize a global tracer 
tracer_provider = init_tracing(project_name="mcp-agent-evaluation", endpoint="http://localhost:6006")
tracer = tracer_provider.get_tracer("eval-runner-tracer")


# ----------------- Comparison helpers -----------------
def floats_close(a: Any, b: Any, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
    if a is None or b is None: return a is None and b is None
    try:
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb): return True
        return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
    except (TypeError, ValueError): return a == b

def compare_results(
    got: Dict[str, Any],
    expected: Dict[str, Any],
    rtol: float = 1e-3,
    atol: float = 1e-6
) -> bool:
    """Compare two dictionaries of numerical simulation results.

    This function provides a robust way to check if two dictionaries,
    representing simulation outputs, are numerically equivalent within a
    specified tolerance. It enforces that all metrics must be numeric
    and that both dictionaries must have the exact same set of keys.

    Parameters
    ----------
    got : dict
        The dictionary of actual results obtained from a simulation run.
        All values are expected to be numeric (int or float).
    expected : dict
        The dictionary of expected results to compare against.
        All values are expected to be numeric (int or float).
    rtol : float, optional
        The relative tolerance parameter for `numpy.allclose`.
        Default is 1e-3.
    atol : float, optional
        The absolute tolerance parameter for `numpy.allclose`.
        Default is 1e-6.

    Returns
    -------
    bool
        True if the dictionaries are a match, False otherwise. A match
        requires that:
        1. Both dictionaries have the identical set of keys.
        2. All values in both dictionaries are numeric.
        3. All corresponding numeric values are close, as determined by
           `numpy.allclose` with the given tolerances.

    Examples
    --------
    >>> got = {'metric_a': 1.0001, 'metric_b': 200.0}
    >>> expected = {'metric_a': 1.0, 'metric_b': 200.5}
    >>> compare_results(got, expected, rtol=1e-2)
    True

    >>> got = {'metric_a': 1.01, 'metric_b': 200.0}
    >>> expected = {'metric_a': 1.0, 'metric_b': 200.0}
    >>> compare_results(got, expected, rtol=1e-3)
    False

    >>> got = {'metric_a': 1.0, 'metric_b': 'fail'}
    >>> expected = {'metric_a': 1.0, 'metric_b': 2.0}
    >>> compare_results(got, expected)
    False

    >>> got = {'metric_a': 1.0}
    >>> expected = {'metric_a': 1.0, 'metric_b': 2.0}
    >>> compare_results(got, expected)
    False
    """
    if expected is None or got is None:
        return False

    s_got = pd.Series(got)
    s_expected = pd.Series(expected)

    # 1. Check for structural differences (different keys).
    if set(s_got.index) != set(s_expected.index):
        return False
    
    # 2. Verify that ALL values in BOTH series are numeric.
    # pd.api.types.is_number is a robust way to check for int/float.
    if not (s_got.apply(pd.api.types.is_number).all() and
            s_expected.apply(pd.api.types.is_number).all()):
        return False

    # 3. Align and compare using NumPy's tolerance-based function.
    # We already checked for key equality, so we can align `expected` to `got`.
    s_expected_aligned = s_expected.loc[s_got.index]

    # `np.allclose` is the gold standard for comparing arrays of floats.
    return np.allclose(
        s_got.values,
        s_expected_aligned.values,
        rtol=rtol,
        atol=atol,
        equal_nan=True  # Considers two NaN values to be equal.
    )

# ----------------- Agent run helpers ----------------
async def run_agent_once(compiled_graph, user_input: str, llm: OllamaLLM) -> AgentState:
    state_in: AgentState = {"user_input": user_input, "retry_count": 0, "validation_history": []}
    return await compiled_graph.ainvoke(state_in)

def extract_sim_result(state: AgentState) -> Optional[Dict[str, Any]]:
    return state.get("simulation_result")

# ---------------- Bulk Ingest Function for Phoenix ----------------
def bulk_ingest_to_phoenix(json_path: str, eval_name: str = "Simulation Agent Eval"):
    """
    Loads an enriched evals.json file and bulk-ingests into Phoenix,
    now including score, label, and explanation columns.
    """
    with open(json_path, "r") as f:
        evals = json.load(f)

    eval_records = []
    for ex_name, case in evals.items():
        context = case.get("context")
        if not context or "span_id" not in context or "trace_id" not in context:
            print(f"⚠️ Skipping '{ex_name}': missing trace/span context in {json_path}")
            continue

        is_passed = bool(case.get("passed"))
        
        # **FIX:** Add all three required columns: score, label, and explanation.
        eval_records.append({
            "context.trace_id": context["trace_id"],
            "context.span_id": context["span_id"],
            "example_id": ex_name,
            "score": 1 if is_passed else 0,
            "label": "Pass" if is_passed else "Fail",
            "explanation": "Agent result matched expected values within tolerance." if is_passed 
                         else "Agent result did not match expected values.",
        })

    if not eval_records:
        print("No valid records found to ingest. Did you run the agent first to generate evals.json?")
        return
        
    eval_df = pd.DataFrame(eval_records)
    eval_df = eval_df.set_index("context.span_id")

    client = px.Client()
    client.log_evaluations(SpanEvaluations(eval_name=eval_name, dataframe=eval_df))
    print(f"[✓] Pushed {len(eval_df)} eval rows to Phoenix under '{eval_name}'")

# ---------------- Main eval runner ----------------
async def run_all_and_save(model_name: str = "gemma3:27b", limit: int = None):
    """
    Runs the full evaluation pipeline and saves an enriched evals.json
    that now includes the necessary trace/span context for Phoenix.
    """
    with open("evals/evals.json", "r") as f:
        evals = json.load(f)

    llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
    compiled_graph = build_graph(llm)

    # Use islice to limit the loop if a limit is provided
    items_to_process = itertools.islice(evals.items(), limit) if limit else evals.items()

    for ex_name, case in items_to_process:
        # **FIX:** Create a parent span for each eval run to capture its context
        with tracer.start_as_current_span(f"eval_run: {ex_name}") as span:
            # Capture the context from the currently active span
            span_context = span.get_span_context()
            trace_id = f"{span_context.trace_id:032x}"
            span_id = f"{span_context.span_id:016x}"

            # Run the agent pipeline
            final_state = await run_agent_once(compiled_graph, case["user_input"], llm)
            
            # Process results
            got = extract_sim_result(final_state)
            passed = compare_results(got, case.get("expected_results"))

            # Store results and the new context back into the dictionary
            case["agent_result"] = got
            case["passed"] = passed
            case["context"] = {"trace_id": trace_id, "span_id": span_id}
            
            # Optionally add attributes to the span
            span.set_attribute("eval.passed", passed)
            span.set_attribute("eval.example_id", ex_name)


    with open("evals/evals_output.json", "w") as f:
        json.dump(evals, f, indent=2)

    print("[✓] Saved enriched evals.json with trace/span context.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run evals and/or bulk-ingest into Phoenix")
    parser.add_argument("--skip-run", action="store_true", help="Skip agent runs and just bulk-ingest existing evals.json")
    parser.add_argument("--eval-name", default="Simulation Agent Eval")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of evaluations to run for debugging.")
    args = parser.parse_args()

    if args.skip_run:
        bulk_ingest_to_phoenix("evals/evals_output.json", eval_name=args.eval_name)
    else:
        asyncio.run(run_all_and_save(model_name="gpt-oss:20b", limit=args.limit))
        bulk_ingest_to_phoenix("evals/evals_output.json", eval_name=args.eval_name)
