# server.py
from fastmcp import FastMCP  # Correct
from model import run_simulation_from_dict

mcp = FastMCP("Call Centre Simulation MCP Server")

# Simulation run tool
@mcp.tool()
def run_experiment(parameters: dict) -> dict:
    """Run one simulation with given JSON-style parameters."""
    return run_simulation_from_dict(parameters)

# JSON schema template to help LLMs construct parameters
@mcp.resource(uri="resource://get_experiment_template")
def get_experiment_template() -> dict:
    """Return a JSON-compatible parameter schema for the simulation."""
    return {
        "n_operators": {"type": "int", "default": 13, "description": "Number of call operators"},
        "n_nurses": {"type": "int", "default": 10, "description": "Number of nurse callbacks"},
        "mean_iat": {"type": "float", "default": 0.6, "description": "Mean time between arrivals"},
        "call_low": {"type": "float", "default": 5.0},
        "call_mode": {"type": "float", "default": 7.0},
        "call_high": {"type": "float", "default": 10.0},
        "callback_prob": {"type": "float", "default": 0.4},
        "nurse_consult_low": {"type": "float", "default": 10.0},
        "nurse_consult_high": {"type": "float", "default": 20.0},
        "random_seed": {"type": "int", "default": 0},
        "run_length": {"type": "int", "default": 1000}
    }

# Text description of the model
@mcp.resource(uri="resource://model_description")
def describe_model() -> str:
    return (
        "This is a discrete-event simulation of a healthcare call centre. "
        "Patients call in, speak with an operator, and ~40% may need a nurse callback. "
        "Calls are processed using SimPy queues and resources. The model tracks wait times, "
        "utilization, and callback rates. You can adjust operator/nurse numbers, call durations, "
        "demand rates, and more. Example: 'Run with 14 operators and 5% extra demand'."
    )

if __name__ == "__main__":
    print("Starting MCP server on port 8000")
    mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")

