# server.py
from fastmcp import FastMCP 
from langchain_core.prompts import PromptTemplate
from fastmcp.prompts.prompt import PromptMessage, TextContent
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
    with open("resources/schema.json") as f:
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

@mcp.prompt(name="parameter_prompt", description="Prompt to turn user input into simulation parameters.")
def parameter_prompt(schema: str, user_input: str) -> PromptMessage:
    """Returns a parameterized prompt template read from file."""
    with open("resources/parameter_prompt.txt", encoding="utf-8") as f:
        prompt_template_text = f.read()

    # Let LangChain handle formatting properly
    prompt = PromptTemplate.from_template(prompt_template_text)
    filled_prompt = prompt.format(schema=schema, user_input=user_input)

    return PromptMessage(
        role="user",
        content=TextContent(type="text", text=filled_prompt)
    )


@mcp.tool()
def validate_parameters(parameters: dict) -> dict:
    """
    Validate simulation parameters against schema.
    Returns {"is_valid": bool, "errors": [str, ...]}..
    """
    with open("resources/schema.json") as f:
        schema = json.load(f)
    errors = []
    for key, value in parameters.items():
        if key not in schema:
            errors.append(f"Unknown parameter: {key}")
            continue
        spec = schema[key]
        # Type check
        expected_type = int if spec["type"] == "int" else float
        if not isinstance(value, expected_type):
            errors.append(f"{key} must be {spec['type']}")
            continue
        # Range check
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{key} below minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{key} above maximum {spec['maximum']}")
    # Inter-parameter logic (optional)
    if all(x in parameters for x in ("call_low", "call_mode", "call_high")):
        if not (parameters["call_low"] <= parameters["call_mode"] <= parameters["call_high"]):
            errors.append("call_low ≤ call_mode ≤ call_high violated")
    if all(x in parameters for x in ("nurse_consult_low", "nurse_consult_high")):
        if not (parameters["nurse_consult_low"] <= parameters["nurse_consult_high"]):
            errors.append("nurse_consult_low ≤ nurse_consult_high violated")
    return {"is_valid": len(errors) == 0, "errors": errors}


if __name__ == "__main__":
    print("Starting MCP server on port 8000")
    mcp.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")

