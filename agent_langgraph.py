# simulation_graph_agent.py
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

console = Console()

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
    simulation_result: Optional[dict]

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
    >>> clean_llm_response("``````")
    '{"key": "value"}'

    >>> clean_llm_response("Some text {\"key\": \"value\"} more text")
    '{"key": "value"}'

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
        # Ask MCP for the ready-made prompt that tells an LLM how to jsonify
        prompt_resp = await cl.get_prompt(
            "parameter_jsonification_prompt",
            {"schema": state["schema"], "user_input": state["user_input"]},
        )
        
    prompt_text = prompt_resp.messages[0].content.text
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]🧠 Reasoning about simulation parameters..."),
        transient=True,  # Removes progress bar after completion
    ) as progress:
        task = progress.add_task("summarising", total=None)
        llm_out = llm.invoke(prompt_text)
        progress.remove_task(task)
    
    state["parameters"] = json.loads(clean_llm_response(llm_out))
    return state


async def validate_parameters(state: Dict[str, Any]) -> Dict[str, Any]:
    async with Client("http://localhost:8001/mcp") as cl:
        resp = await cl.call_tool(
            "validate_simulation_parameters",
            {"parameters": state["parameters"]},
        )
    state["validation"] = resp.data
    return state


def validation_branch(state: Dict[str, Any]) -> str:
    return "valid" if state["validation"]["is_valid"] else "invalid"


async def run_simulation(state: Dict[str, Any]) -> Dict[str, Any]:
    async with Client("http://localhost:8001/mcp") as cl:
        resp = await cl.call_tool(
            "run_call_centre_simulation",
            {"parameters": state["parameters"]},
        )
    state["simulation_result"] = resp.data
    return state


async def summarise_parameters(state: Dict[str, Any], llm: OllamaLLM) -> Dict[str, Any]:
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]✏️ Summarising parameters used..."),
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

    # 2. create entry point and edges
    graph.set_entry_point("get_schema")
    graph.add_edge("get_schema", "jsonify")
    graph.add_edge("jsonify", "validate")
    graph.add_conditional_edges(
        "validate",
        validation_branch,
        {"valid": "format_params", "invalid": END}
    )
    graph.add_edge("format_params", "run_sim")
    graph.add_edge("run_sim", END)
    
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
    final_state = await compiled_graph.ainvoke({"user_input": user_request})

    # 4. Report results
    console.rule("[bold green]RESULTS")
    if "simulation_result" in final_state:
        display_param_summary_table(final_state)
        display_results_table(final_state)
    else:
        console.print("[red]❌ Parameter validation failed")
        console.print_json(data=final_state["validation"])

if __name__ == "__main__":
    model_name = "gemma3:27b"
    asyncio.run(main(model_name))



