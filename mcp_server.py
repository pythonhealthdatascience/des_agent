# server.py
from fastmcp import FastMCP  # Correct
from model import run_simulation_from_dict
import json

mcp = FastMCP("Call Centre Simulation MCP Server")

# Simulation run tool
@mcp.tool()
def run_experiment(parameters: dict) -> dict:
    """Run one simulation with given JSON-style parameters."""
    return run_simulation_from_dict(parameters)

# JSON schema template to help LLMs construct parameters
@mcp.resource(uri="resource://get_experiment_template")
def get_experiment_template() -> dict:
    with open("schema.json") as f:
        return json.load(f)

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
    mcp.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")

