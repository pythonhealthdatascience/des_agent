"""
Call Centre Simulation MCP Server
=================================

This module implements the Model Content Protocol (MCP) server for a discrete-event 
simulation (DES) model of a healthcare call centre. The server exposes executable 
simulation tools, parameter validation utilities, schema/documentation resources, 
and prompting interfaces designed for integration with language model (LLM) agents.

Main Features
-------------
- Run call centre simulations with configurable staffing and demand parameters.
- Validate proposed simulation parameters against a machine-readable JSON schema.
- Provide self-describing resources (schema, natural language model documentation).
- Generate LLM prompts to map user requests onto structured simulation parameter sets.

Accessible Functionalities
--------------------------
- `run_call_centre_simulation`: Simulate the healthcare call centre and return metrics.
- `validate_simulation_parameters`: Pre-validate parameter sets to prevent runtime errors.
- `get_experiment_parameter_schema`: Obtain the schema for all simulation parameters.
- `get_model_description`: Retrieve a human-oriented description of the simulation model.
- `parameter_jsonification_prompt`: Produce a prompt for LLMs to convert text requests into structured parameters.

Intended Usage
--------------
This server is intended to be used with agentic or LLM-driven client interfaces, 
providing a robust, schema-driven, and discoverable API for advanced simulation reasoning, 
configuration generation, and automated experimentation.

"""

from fastmcp import FastMCP 
from langchain_core.prompts import PromptTemplate
from fastmcp.prompts.prompt import PromptMessage, TextContent
from model import run_simulation_from_dict
import json
import re

mcp = FastMCP("Call Centre Simulation MCP Server")

@mcp.tool(
    name="run_call_centre_simulation",
    description="""
Runs a discrete-event healthcare call centre simulation with specified parameters, returning performance metrics.

Inputs: parameters (dict) — JSON object matching the experiment schema.
Example: {"operators": 12, "nurses": 5, "call_arrival_rate": 120}

Returns: dict with simulation metrics, such as mean wait times and resource utilizations.

Tags: ["simulation", "call_centre", "experiment"]
""")
def run_call_centre_simulation(parameters: dict) -> dict:
    """
    Run a discrete-event healthcare call centre simulation.

    Parameters
    ----------
    parameters : dict
        Simulation configuration as a JSON object matching the experiment parameter schema.
        Example:
            {"operators": 14, "nurses": 4, "call_arrival_rate": 120}

    Returns
    -------
    dict
        Simulation results, such as:
        {
            "mean_wait_time": float,
            "operator_utilization": float,
            "callback_rate": float,
            ...
        }
    """
    return run_simulation_from_dict(parameters)

@mcp.resource(
    uri="resource://schema/experiment_parameters",
    description="""
Returns the JSON schema defining all allowed input parameters, parameter types, and value constraints.

Outputs: dict (JSON schema), sent as a JSON object.

Tags: ["schema", "parameters", "template"]
""")
def get_experiment_parameter_schema() -> dict:
    """
    Retrieve the experiment parameter JSON schema.

    Returns
    -------
    dict
        JSON schema describing all allowable simulation parameters, accepted types, and value constraints.
        Example:
            {
                "operators": {"type": "int", "minimum": 1, "maximum": 50},
                ...
            }
    """
    with open("resources/schema.json") as f:
        return json.load(f)

@mcp.resource(
    uri="resource://model/description",
    description="""
Provides a natural language description of the healthcare call centre simulation model.

Outputs: str (text description).

Tags: ["model", "description", "documentation"]
""")
def get_model_description() -> str:
    """
    Get a natural language description of the call centre simulation model.

    Returns
    -------
    str
        Human-readable description of the model's structure, main features, and configurable aspects.
    """
    return (
        "This is a discrete-event simulation of a healthcare call centre. "
        "Patients call in, interact with operators, and a subset may require a nurse callback. "
        "Simulation components: SimPy queues and resources. Tracks wait times, utilization, and callback rates. "
        "Configurable parameters: number of operators and nurses, call durations and rates, etc. "
        "Sample: 'Run with 14 operators and 5% higher demand.'"
    )

@mcp.prompt(
    name="parameter_jsonification_prompt",
    description="""
INSTRUCTION TO LLM: Convert a user's freeform simulation request into a JSON object matching a given schema.

Inputs:
- schema (str): JSON Schema as a string
- user_input (str): User's natural language request

Returns: PromptMessage (LLM input) guiding the agent to produce valid JSON parameters.

Tags: ["jsonification", "schema_mapping", "prompt", "parameters"]
""")
def parameter_jsonification_prompt(schema: str, user_input: str) -> PromptMessage:
    """
    Create an LLM prompt for mapping a user's request to a simulation parameter JSON.

    Parameters
    ----------
    schema : str
        JSON schema (as a string) describing expected parameters.
    user_input : str
        User's freeform simulation request.

    Returns
    -------
    PromptMessage
        Prompt for LLM to generate a valid parameters JSON according to the schema.
    """
    with open("resources/parameter_prompt.txt", encoding="utf-8") as f:
        prompt_template_text = f.read()
    prompt = PromptTemplate.from_template(prompt_template_text)
    filled_prompt = prompt.format(schema=schema, user_input=user_input)
    return PromptMessage(
        role="user",
        content=TextContent(type="text", text=filled_prompt)
    )

@mcp.tool(
    name="validate_simulation_parameters",
    description="""
Validate a proposed set of simulation parameters (JSON object) against the experiment schema.

Inputs: parameters (dict)
Returns: {"is_valid": bool, "errors": [str, ...]} — status and explanation.

Tags: ["validation", "parameter_check", "pre_run_check", "schema"]
""")
def validate_simulation_parameters(parameters: dict) -> dict:
    """
    Validate simulation parameters against the experiment schema.

    Parameters
    ----------
    parameters : dict
        Proposed parameter set for simulation, as a JSON-compatible dictionary.

    Returns
    -------
    dict
        {
            "is_valid": bool,
            "errors": list of str
        }
        is_valid: True if all parameters are recognized and within allowed ranges.
        errors: Descriptive messages for each invalid parameter or failed interparameter constraint.
    """
    with open("resources/schema.json") as f:
        schema = json.load(f)
    errors = []
    for key, value in parameters.items():
        if key not in schema:
            errors.append(f"Unknown parameter: {key}")
            continue
        spec = schema[key]
        expected_type = int if spec["type"] == "int" else float
        if not isinstance(value, expected_type):
            errors.append(f"{key} must be {spec['type']}")
            continue
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{key} below minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{key} above maximum {spec['maximum']}")
    if all(x in parameters for x in ("call_low", "call_mode", "call_high")):
        if not (parameters["call_low"] <= parameters["call_mode"] <= parameters["call_high"]):
            errors.append("call_low ≤ call_mode ≤ call_high violated")
    if all(x in parameters for x in ("nurse_consult_low", "nurse_consult_high")):
        if not (parameters["nurse_consult_low"] <= parameters["nurse_consult_high"]):
            errors.append("nurse_consult_low ≤ nurse_consult_high violated")
    return {"is_valid": len(errors) == 0, "errors": errors}

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")
