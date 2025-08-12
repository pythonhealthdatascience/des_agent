"""
Agent Self-Reflection Simulation Parameter Generator

This module implements an simple agent that generates and validates simulation 
parameters using natural language input and self-reflection capabilities. The agent 
employs a graph-based workflow to iteratively refine parameters until they meet 
validation requirements.

Overview
--------
The agent uses a state machine architecture built with LangGraph to orchestrate 
the parameter generation process. When validation fails, the agent reflects on 
the errors and attempts to correct them through multiple retry cycles.

Core Workflow
-------------
1. Schema Retrieval: Fetch parameter schema from MCP server
2. Parameter Generation: Use LLM to convert natural language to JSON parameters  
3. Validation: Validate parameters against schema constraints
4. Self-Reflection: On validation failure, analyze errors and retry
5. Simulation Execution: Run simulation with validated parameters
6. Results Display: Present formatted results and parameter summaries

Usage
-----
Run from command line with optional model specification:

    $ python agent_self_reflection.py --llm gemma3:27b
    $ python agent_self_reflection.py --llm mistral:7b

The agent will prompt for a natural language simulation description and 
automatically handle parameter generation, validation, and execution.

Hard constraints
-------------
MAX_RETRIES : int
    Maximum number of parameter generation attempts before bailout (default: mam4)

Examples
--------
Natural language inputs the agent can process:
- "Simulate 14 operators, 12 nurses and 5% extra demand"
- "Run scenario with high staffing and normal call volume"
- "Test configuration with minimal staff"

Notes
-----
Requires running MCP server on localhost:8001 and Ollama server on localhost:11434.
Different LLM models show varying performance - gemma3:27b and mistral:7b are 
recommended for reliable parameter generation.

"""

import asyncio, json, re
from typing import Dict, Any, Optional, TypedDict

from fastmcp import Client
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.markdown import Markdown

from functools import partial

import pandas as pd

import argparse

console = Console()

# maximum tries are jsonifying parameter list...
MAX_RETRIES = 4

PARAMETER_TABLE_TEMPLATE = """
Given the following JSON object representing parameters updated in a simulation model:

{json_object}

Task:  
Convert this JSON object into a Markdown table with two columns: **Parameter** and **Value**. 
Each key should map to its value on a separate row. 
Display only the table in clean Markdown format.
Do NOT include any text before the table such as 'Here is the output:'
Please add a graph emoji to the parameter column header.

Output

**Parameters used in simulation**

[markdown table]
"""


class AgentState(TypedDict):
    user_input: str
    schema: dict
    parameters: dict
    formatted_parameters: str
    validation: Optional[dict]
    validation_history: list 
    simulation_result: Optional[dict]
    retry_count: int  

# --------------------------- Helper funcs ---------------------------------------- #
def clean_llm_response(response: Optional[str]) -> str:
    """
    Clean LLM response to extract JSON from markdown code blocks.

    Large Language Models often wrap JSON responses in markdown code blocks.
    This function removes markdown formatting and extracts the JSON content.

    Parameters
    ----------
    response : Optional[str]
        Raw LLM response that may contain markdown formatting

    Returns
    -------
    str
        Cleaned JSON string ready for parsing

    Examples
    --------
    >>> clean_llm_response("```{'key':'value'}```")
    "{'key': 'value'}"

    >>> clean_llm_response("```json{'key':'value'}```")
    "{'key': 'value'}"

    >>> clean_llm_response("Some text {'key':value'} more text")
    "{'key': 'value'}"

    >>> clean_llm_response("")
    '{}'
    """
    if not response:
        return "{}"

    # Remove markdown code block markers
    response = response.strip()

    # Remove ```json and ```
    if response.startswith("```json"):
        response = response[7:]  # Remove ```
    if response.startswith("```"):
        response = response[3:]  # Remove ```
    if response.endswith("```"):
        response = response[:-3]  # Remove closing ```

    # Find JSON object boundaries
    response = response.strip()
    start_idx = response.find("{")
    end_idx = response.rfind("}")

    if start_idx != -1 and end_idx != -1:
        return response[start_idx : end_idx + 1]

    return response




# --------------------------- graph nodes ------------------------------------ #
async def fetch_schema(state: Dict[str, Any]) -> Dict[str, Any]:
    async with Client("http://localhost:8001/mcp") as cl:
        # Resource URI exactly as exposed by your server
        res = await cl.read_resource("resource://schema/experiment_parameters")
        state["schema"] = res[0].text if hasattr(res[0], "text") else res[0]
    return state


async def generate_parameters(state: Dict[str, Any], llm: OllamaLLM) -> Dict[str, Any]:
    async with Client("http://localhost:8001/mcp") as cl:

        # prompt parameters
        prompt_vars = {
            "schema": state["schema"],
            "user_input": state["user_input"]
        }
        # include validation errors if present in state memory
        validation_errors = state.get("validation", {}).get("errors")
        if validation_errors:
            prompt_vars["validation_errors"] = "\n".join(f"- {e}" for e in validation_errors)
        else:
            prompt_vars["validation_errors"] = ""  # always supply a string

        # Ask MCP for the ready-made prompt that tells an LLM how to jsonify
        prompt_resp = await cl.get_prompt("parameter_jsonification_prompt", prompt_vars)
       
    prompt_text = prompt_resp.messages[0].content.text
    
    progress_text = "[bold green]🧠 Reasoning about simulation parameters."
    if state["retry_count"] > 0:
        progress_text += f"[RETRY {state['retry_count']}]"
    with Progress(
        SpinnerColumn(),
        TextColumn(progress_text),
        transient=True,  
    ) as progress:
        task = progress.add_task("summarising", total=None)
        llm_out = llm.invoke(prompt_text)
        progress.remove_task(task)
    
    cleaned_response = clean_llm_response(llm_out)

    try:
        state["parameters"] = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        # Force validation failure by setting invalid parameters
        # and create a mock validation response for self-reflection
        state["parameters"] = {"__json_parse_error__": True}
        console.print(f"[yellow]⚠️ JSON parsing failed - will retry with feedback")
    except Exception as e:
        # Handle other unexpected errors similarly
        state["parameters"] = {"__unexpected_error__": True}
        console.print(f"[yellow]⚠️ Unexpected error - will retry")

    return state


async def validate_parameters(state: Dict[str, Any]) -> Dict[str, Any]:
    async with Client("http://localhost:8001/mcp") as cl:
        resp = await cl.call_tool(
            "validate_simulation_parameters",
            {"parameters": state["parameters"]},
        )

    # Add current validation to history before overwriting
    if "validation_history" not in state:
        state["validation_history"] = []
    
    # Store the validation attempt with context
    state["validation_history"].append({
        "attempt": state.get("retry_count", 0) + 1,
        "parameters": state["parameters"].copy(),
        "validation_result": resp.data.copy()
    })

    state["validation"] = resp.data
    return state

def validation_branch(state: Dict[str, Any]) -> str:
    return "valid" if state["validation"]["is_valid"] else "invalid"

def retry_branch(state: AgentState) -> str:
    """routing node.  bail out if max retries exceeded otherwise jsonify"""
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "bail_out"
    return "jsonify"

def increment_retry(state: AgentState) -> AgentState:
    """Increment the number of attempts at jsonifying parameters"""
    state["retry_count"] = state.get("retry_count", 0) + 1
    report_parameter_reflection_failure(state, MAX_RETRIES)
    return state

def bail_out_node(state: AgentState) -> AgentState:
    """A bail out node if exceeded max attempts at jsonifying parameters"""
    state["error"] = "Maximum retries exceeded during parameter reflection."
    return state

async def run_simulation(state: Dict[str, Any]) -> Dict[str, Any]:
    async with Client("http://localhost:8001/mcp") as cl:
        resp = await cl.call_tool(
            "run_call_centre_simulation",
            {"parameters": state["parameters"]},
        )
    state["simulation_result"] = resp.data
    return state


async def summarise_parameters(state: Dict[str, Any], llm: OllamaLLM) -> Dict[str, Any]:
    """Generates a formatted markdown table of parameters from JSON. 
    Could do this programatically, but just for fun we will use a LLM"""
    progress_text = "[bold blue]✏️ Summarising parameters used..."
    with Progress(
        SpinnerColumn(),
        TextColumn(progress_text),
        transient=True,  
    ) as progress:
        task = progress.add_task("summarising_params", total=None)
        prompt = PromptTemplate.from_template(PARAMETER_TABLE_TEMPLATE)
        param_display_prompt = prompt.format(json_object=state["parameters"])
        llm_result = llm.invoke(param_display_prompt)
        progress.remove_task(task)

    state["formatted_parameters"] = llm_result
    return state

# --------------------------- build the graph -------------------------------- #
def build_graph(llm: OllamaLLM) -> StateGraph:
    graph = StateGraph(AgentState)  # Using the TypedDict from above

    # 1. create notes
    # NB: I've used a partial functions here to solve the issue with passing extra parameters to a coroutine
    graph.add_node("get_schema", fetch_schema)
    graph.add_node("jsonify", partial(generate_parameters, llm=llm))
    graph.add_node("validate", validate_parameters)
    graph.add_node("run_sim", run_simulation)
    graph.add_node("format_params", partial(summarise_parameters, llm=llm))
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("bail_out", bail_out_node)

    # 2. create entry point and edges
    graph.set_entry_point("get_schema")
    graph.add_edge("get_schema", "jsonify")
    graph.add_edge("jsonify", "validate")
    graph.add_conditional_edges(
        "validate",
        validation_branch,
        {"valid": "format_params", "invalid": "increment_retry"}
    )

    # handles limited number of retrries. links to bail_out (and END) 
    # if retries exceeds a hard limit.
    graph.add_conditional_edges(
        "increment_retry",
        retry_branch,
        {"jsonify": "jsonify", "bail_out": "bail_out"}
    )  
    
    graph.add_edge("format_params", "run_sim")
    graph.add_edge("run_sim", END)
    graph.add_edge("bail_out", END)
    
    return graph.compile()

def display_results_table(state: AgentState):
    """
    Display simulation results to the user
    """
    console.print(Markdown("✅ **Simulation complete.**"))
    df = pd.DataFrame(
        list(state["simulation_result"].items()), columns=["KPIs", "Values"]
    )
    console.print(Markdown(df.round(2).to_markdown(index=False)))


def display_param_summary_table(state: AgentState):
    """
    Display a summary table of parameters for human-in-loop validation
    """
    console.print(Markdown(state["formatted_parameters"]))

def report_parameter_reflection_failure(state: dict, max_retries: int):
    """
    Display a clear error message for parameter reflection failure
    """
    retry_count = state.get("retry_count", 0)
    console.print(
        f"[bold red]❌ Parameter validation failed after {retry_count} attempt{'s' if retry_count!=1 else ''}."
    )
    if "validation" in state and "errors" in state["validation"]:
        errors = state["validation"]["errors"]
        if errors:
            console.print("[red]Last validation errors were:")
            for err in errors:
                console.print(f"[red]- {err}")
        else:
            console.print("[red]No specific validation errors were provided by the server.")
    else:
        console.print("[red]No validation error details are available.")
        console.print(f"[yellow]The agent was unable to generate valid simulation parameters in {retry_count} tries (limit: {max_retries}).")
        console.print("[yellow]Try rephrasing your request or ensure parameter values are within allowed ranges. Refer to the simulation parameter schema for guidance.")


def display_validation_history(state: AgentState):
    """Display the history of validation attempts"""
    history = state.get("validation_history", [])
    
    if not history:
        return
        
    console.print(Markdown("🔍 **Parameter Generation Issues**"))
    
    for entry in history:
        attempt_num = entry["attempt"]
        is_valid = entry["validation_result"]["is_valid"]
        errors = entry["validation_result"]["errors"]
        
        status = "✅ Success" if is_valid else "❌ Failed"
        console.print(f"\n**Attempt {attempt_num}:** {status}")
        
        if errors:
            console.print("Errors encountered:")
            for error in errors:
                console.print(f"{error}")



async def main(model_name: str) -> None:

    # 1. Setup the graph and LLM
    llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
    compiled_graph = build_graph(llm)
    
    # 2. Prompt user
    user_request = Prompt.ask(
        "Simulation request",
        default="Simulate 14 operators, 12 nurses and 5 % extra demand"
    )

    # 3. invoke graph
    final_state = await compiled_graph.ainvoke({
        "user_input": user_request,
        "retry_count": 0,     
        "validation_history": []
    })

    # 4. Report results
    console.rule("[bold green]RESULTS")
    if "simulation_result" in final_state:
        display_param_summary_table(final_state)
        display_results_table(final_state)
    
    retry_count = final_state.get("retry_count", 0)
    if retry_count > 0:
        display_validation_history(final_state)

def parse_arguments():
    """Parse command line arguments for model selection."""
    parser = argparse.ArgumentParser(
        description="Simulation Agent Workflow - Experiment with a simulation model using natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python agent_self_reasoning.py -l gemma3:27b
            python agent_self_reasoning.py -l mistral:7b 
            """,
    )

    parser.add_argument(
        "-l",
        "--llm",
        type=str,
        default="gemma3:27b",
        help="Model to use for generating parameters (default:gemma3:27b)",
    )

    return parser.parse_args()


if __name__ == "__main__":

    # Parse command line arguments
    args = parse_arguments()

    asyncio.run(main(model_name=args.llm))



