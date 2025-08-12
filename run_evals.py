import asyncio
import json
import math
from typing import Dict, Any, Tuple, Optional

# Import the agent builder and its dependencies from your agent module
# Assumes agent_self_reflection.py exposes `build_graph`, `AgentState`, and a factory for the LLM.
from agent_self_reflection import build_graph, AgentState
from langchain_ollama import OllamaLLM


def floats_close(a: Any, b: Any, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
    """
    Compare numbers with tolerance, pass-through for non-numeric equality.
    Returns True if both are NaN, or within tolerances for floats, or exactly equal otherwise.
    """
    # Handle None
    if a is None or b is None:
        return a is None and b is None

    # Try numeric compare
    try:
        fa = float(a)
        fb = float(b)
        # Handle NaNs
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
    except (TypeError, ValueError):
        # Non-numeric: exact equality
        return a == b


def compare_results(
    got: Dict[str, Any],
    expected: Dict[str, Any],
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compare simulation outputs to expected_results with numeric tolerance.
    Returns a dict with per-key comparison, diffs, and overall pass flag.
    """
    keys = sorted(set(got.keys()) | set(expected.keys()))
    per_key = {}
    all_pass = True

    for k in keys:
        g = got.get(k, None)
        e = expected.get(k, None)
        ok = floats_close(g, e, rtol=rtol, atol=atol)
        if not ok:
            all_pass = False
        per_key[k] = {
            "expected": e,
            "got": g,
            "match": ok,
        }

    return {
        "pass": all_pass,
        "details": per_key,
        "rtol": rtol,
        "atol": atol,
    }


async def run_agent_once(
    compiled_graph,
    user_input: str,
    llm: OllamaLLM,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    Runs the agent end-to-end for a single natural-language user_input.
    Returns the final state, including simulation_result or error.
    """
    state_in: AgentState = {
        "user_input": user_input,
        "retry_count": 0,
        "validation_history": [],
    }
    final_state: AgentState = await compiled_graph.ainvoke(state_in)
    return final_state


def extract_agent_simulation_result(final_state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract the simulation_result if present and return a compact agent_run summary
    useful for evaluation logging.
    """
    sim_res = final_state.get("simulation_result")
    agent_run = {
        "retry_count": final_state.get("retry_count", 0),
        "had_bailout": final_state.get("error") == "Maximum retries exceeded during parameter reflection.",
        "validation_history": final_state.get("validation_history", []),
        "parameters": final_state.get("parameters", {}),
        "formatted_parameters": final_state.get("formatted_parameters", None),
    }
    return sim_res, agent_run


async def main(
    input_json_path: str = "evals/evals.json",
    output_json_path: str = "evals/evals_output.json",
    model_name: str = "gemma3:27b",
    llm_base_url: str = "http://localhost:11434",
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> None:
    # 1) Load evals
    with open(input_json_path, "r") as f:
        evals = json.load(f)

    # 2) Build LLM and graph once
    llm = OllamaLLM(model=model_name, base_url=llm_base_url)
    compiled_graph = build_graph(llm)

    # 3) Run each eval in a loop via the agent
    for key, case in evals.items():
        user_input = case.get("user_input", "")
        expected = case.get("parameters", {}).get("expected_results") or case.get("expected_results")
        # In your earlier structure, expected_results is at the top level of each example after enrichment.
        # If not present yet, this will remain None and comparison will be skipped.

        final_state = await run_agent_once(compiled_graph, user_input, llm)
        sim_res, agent_run = extract_agent_simulation_result(final_state)

        # Record agent run outputs
        case["agent_run"] = {
            "simulation_result": sim_res,
            "meta": agent_run,
        }

        # 4) Compare to expected_results if available
        if expected is not None and sim_res is not None:
            cmp = compare_results(sim_res, expected, rtol=rtol, atol=atol)
            case["comparison"] = cmp
            case["passed"] = bool(cmp["pass"])
        else:
            case["comparison"] = {
                "note": "Either expected_results or agent simulation_result missing; comparison skipped."
            }
            case["passed"] = False if expected is not None else None

    # 5) Save enriched evals to evals.json
    with open(output_json_path, "w") as f:
        json.dump(evals, f, indent=2)

    print(f"Wrote evaluation results to {output_json_path}")


if __name__ == "__main__":
    # For CLI usage:
    #   python eval_runner.py
    # Optional: parameterize via env vars or argparse if desired.
    asyncio.run(main())
