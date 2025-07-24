"""
Simulation Agent Workflow Module

This module provides an intelligent agent workflow for configuring and running simulation models
using a sophisticated task planning approach. The agent employs dual LLM (Large Language Model)
architecture to create execution plans, generate simulation parameters, and summarize results
automatically.

Key Features:
- Task planning with LLM-generated step-by-step execution plans
- Dual LLM architecture (separate reasoning and summarizing models)
- Command line interface with model selection (--reasoning/-r, --summary/-s)
- FastMCP client integration for dynamic server communication
- Rich progress indicators and formatted console output
- JSON response cleaning and validation
- Dynamic plan execution with memory storage between steps
- Automatic parameter generation, validation, and simulation execution

The workflow follows these steps:
1. User provides a natural language request via command line interface
2. Agent connects to specified LLM models (reasoning and summary)
3. Agent fetches available tools, resources, and prompts from MCP server
4. Reasoning LLM generates a detailed step-by-step execution plan
5. Agent executes the plan dynamically (resource queries, parameter generation, simulation)
6. Summary LLM formats parameters and results are displayed in structured tables

Architecture:
- Planning Model: Handles task decomposition and execution planning
- Summary Model: Formats parameters and results for display
- MCP Integration: Dynamic discovery and execution of server capabilities
- Memory System: Maintains state between execution steps

Examples:
    Run with default models:
    >>> python agent_planning_workflow.py

    Run with custom models:
    >>> python agent_planning_workflow.py -r llama3:latest -s gemma3:27b
    >>> python agent_planning_workflow.py --reasoning deepseek-r1:32b --summary llama3.1:8b

    Interactive usage:
    # User input: "Run with 14 operators and 12 nurses"
    # Agent automatically plans, configures and runs the simulation
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from fastmcp import Client

import asyncio
import json
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Union

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.markdown import Markdown

# prompt for terminal - nothing to do with LLM
from rich.prompt import Prompt
from rich.logging import RichHandler

import argparse
import logging
import os


PARAMETER_TABLE_TEMPLATE = """
Given the following JSON object representing parameters updated in a simulation model:

{json_object}

Task:  
Convert this JSON object into a Markdown table with two columns: **Parameter** and **Value**. 
Each key should map to its value on a separate row. 
Display only the table in clean Markdown format.
Do NOT include any text before the table such as 'Here is the output:'
Please add suitable emoji's to the parameter column header.

Output

**Parameters used in simulation**
[markdown table]
"""

TASK_PLANNING_PROMPT_TEMPLATE = (
    "You are an intelligent assistant that fulfills user simulation requests using MCP server capabilities.\n\n"
    "Available tools:\n{tools}\n\n"
    "Available resources:\n{resources}\n\n"
    "Available prompts:\n{prompts}\n\n"
    "User query:\n{user_input}\n\n"
    "Develop a step-by-step plan to fulfill the user request, using the tools, resources, and prompts listed above.\n\n"
    "Instructions:\n"
    "- For each step:\n"
    "    - Clearly state the action to perform.\n"
    "    - Explicitly indicate which capability is being used: resource, tool, or prompt (and its name).\n"
    "    - Do NOT reference specific parameter names, fields, or structures unless you have already retrieved them from the appropriate resource.\n"
    "    - Do NOT include implicit steps. Only make use of tools, resource and prompts available on the server."
    "    - If a step requires information not yet obtained (such as parameter names/types), plan to retrieve it first.\n"
    "    - Always include the word Step when enumerating steps e.g. Step 1:\n"
    "    - Briefly explain the purpose of each step.\n\n"
    "Output Format:\n"
    "Return your plan as a numbered list. For each step, use the following structure:\n\n"
    "Step N:\n"
    "- Action: [Clear description of what you will do in this step]\n"
    "- Type: [Resource/Tool/Prompt]\n"
    "- Name: [Exact name from available list]\n"
    "- Rationale: [Brief explanation of why this step is necessary]\n"
)


def setup_logging(debug_mode: bool = False):
    """Configure logging to show debug information for the agent."""
    if debug_mode:
        # Create a custom logger for our application only
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Create a simple console handler without Rich formatting noise
        handler = logging.StreamHandler()
        formatter = logging.Formatter("🐛 %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Suppress noisy third-party loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        logger.debug("Debug mode enabled - showing workflow details")
        return logger
    else:
        # Suppress all debug logging in normal mode
        logging.getLogger().setLevel(logging.WARNING)
        return logging.getLogger(__name__)


def debug_print_available_features(features: dict, logger):
    """Print available MCP features in debug mode."""
    logger.debug("=== MCP SERVER CAPABILITIES ===")

    tools = [t.name for t in features.get("tools", [])]
    resources = [r.name for r in features.get("resources", [])]
    prompts = [p.name for p in features.get("prompts", [])]

    logger.debug(f"🔧 Available tools ({len(tools)}): {', '.join(tools)}")
    logger.debug(
        f"📚 Available resources ({len(resources)}): {', '.join(resources)}"
    )
    logger.debug(
        f"📝 Available prompts ({len(prompts)}): {', '.join(prompts)}"
    )


def debug_print_plan(plan_steps: List[Dict[str, str]], logger):
    """Print the LLM-generated plan in debug mode."""
    logger.debug("=== LLM GENERATED PLAN ===")
    for i, step in enumerate(plan_steps, 1):
        logger.debug(f"Step {i}: {step.get('action', 'N/A')}")
        logger.debug(
            f"  → Type: {step.get('type', 'N/A')}, Name: {step.get('name', 'N/A')}"
        )


def format_prompt_options(prompt_list: List[Any], short: bool = False) -> str:
    """
    Convert MCP prompts into numbered or named options for LLM display.

    This function formats a list of prompt objects into a human-readable string
    that can be used by the LLM for prompt selection. Each prompt is displayed
    with its name and description.

    Parameters
    ----------
    prompt_list : List[Any]
        List of prompt objects containing name and description attributes
    short : bool, optional
        If True, displays prompts without numbering (default: False)

    Returns
    -------
    str
        Formatted string containing all prompts with their descriptions

    Examples
    --------
    >>> prompts = [MockPrompt("config_sim", "Configure simulation parameters")]
    >>> format_prompt_options(prompts)
    "1. config_sim: Configure simulation parameters"

    >>> format_prompt_options(prompts, short=True)
    "config_sim: Configure simulation parameters"
    """
    formatted = []
    for i, p in enumerate(prompt_list, 1):
        line = (
            f"{i}. {p.name}: {p.description}"
            if not short
            else f"{p.name}: {p.description}"
        )
        formatted.append(line)
    return "\n".join(formatted)


async def fetch_all_features() -> dict:
    """Fetch all tools, resources, and prompts from the MCP server asynchronously."""
    async with Client("http://localhost:8001/mcp") as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
    return {"tools": tools, "resources": resources, "prompts": prompts}


async def create_task_planning_prompt(user_input: str) -> str:
    """
    Assemble a step-planning prompt for the agent, embedding available tools, resources, prompts, and their descriptions.
    """

    features = await fetch_all_features()

    def format_feature(item):
        name = getattr(item, "name", str(item))
        description = getattr(item, "description", "No description")
        return f"{name}: {description}"

    tool_infos = [format_feature(t) for t in features.get("tools", [])]
    resource_infos = [format_feature(r) for r in features.get("resources", [])]
    prompt_infos = [format_feature(p) for p in features.get("prompts", [])]

    prompt = PromptTemplate.from_template(TASK_PLANNING_PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(
        tools="\n".join(tool_infos) if tool_infos else "None",
        resources="\n".join(resource_infos) if resource_infos else "None",
        prompts="\n".join(prompt_infos) if prompt_infos else "None",
        user_input=user_input,
    )
    return formatted_prompt


async def fetch_all_feature_maps() -> dict:
    features = await fetch_all_features()
    return {
        "tool_map": {t.name: t for t in features.get("tools", [])},
        "resource_map": {r.name: r for r in features.get("resources", [])},
        "prompt_map": {p.name: p for p in features.get("prompts", [])},
        "features": features,  # optionally include the full raw info
    }


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


def parse_llm_plan(plan_text: str):
    """
    Converts a step-by-step plan output from an LLM into a list of dicts.
    """
    # Split by step (supports "Step N" or "Step N:")
    steps = re.split(r"\s*Step \d+:", plan_text)
    steps = [s.strip() for s in steps if s.strip()]  # Remove empties

    plan = []
    item_regex = {
        "action": r"- Action:\s*(.*)",
        "type": r"- Type:\s*(.*)",
        "name": r"- Name:\s*(.*)",
        "rationale": r"- Rationale:\s*(.*)",
    }

    for step in steps:
        entry = {}
        for key, regex in item_regex.items():
            match = re.search(regex, step, flags=re.IGNORECASE)
            if match:
                entry[key] = match.group(1).strip()
        if entry:
            plan.append(entry)
    return plan


async def run_plan(
    plan_steps: List[Dict[str, str]],
    features: Dict[str, List[Any]],
    llm: Any,
    user_input: str,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    Execute a stepwise LLM-generated simulation workflow plan.

    Runs a sequence of planning steps (including resource queries, tool invocations,
    and prompt-driven parameter generation) to fulfill a user simulation request.
    Each step in the plan is dynamically resolved using the available MCP server features.

    Parameters
    ----------
    plan_steps : List[dict]
        List of plan step dictionaries as returned by `parse_llm_plan`.
        Each dictionary should include 'action', 'type', 'name', and 'rationale' keys.
    features : dict
        Dictionary of MCP features as returned by `fetch_all_features()`, containing
        'tools', 'resources', and 'prompts' lists. Each entity must have at least a `.name` attribute.
    llm : Any
        Instantiated LLM (e.g., OllamaLLM) used for prompt-based parameter generation.
    user_input : str
        Original user simulation request in natural language. Passed into prompt steps.

    Returns
    -------
    memory : dict
        Dictionary containing all intermediate and final step results, including:
        - 'parameters' : dict
            Simulation parameters generated by LLM.
        - 'simulation_result' : dict
            Simulation output returned by server tools.
        - ...other step results (by step name).

    Raises
    ------
    KeyError
        If a step name is not found within corresponding feature maps.
    RuntimeError
        If a step's type is not one of ['resource', 'tool', 'prompt'].
    json.JSONDecodeError
        If LLM parameter output cannot be parsed as JSON.

    Examples
    --------
    >>> plan = [{'action': 'Get schema', 'type': 'resource', 'name': 'get_experiment_parameter_schema', ...},
    ...         {'action': 'Generate parameters', 'type': 'prompt', 'name': 'config_sim', ...},
    ...         {'action': 'Run simulation', 'type': 'tool', 'name': 'run_call_centre_simulation', ...}]
    >>> features = await fetch_all_features()
    >>> llm = OllamaLLM(model="llama3:latest", base_url="http://localhost:11434")
    >>> memory = await run_plan(plan, features, llm, "Simulate 10 staff")
    >>> print(memory['simulation_result'])
    {'throughput': 184, 'utilization': 0.89, ...}
    """

    memory = {"user_input": user_input}

    # Index features by name for lookup
    tool_map = {t.name: t for t in features["tools"]}
    resource_map = {r.name: r for r in features["resources"]}
    prompt_map = {p.name: p for p in features["prompts"]}

    async with Client("http://localhost:8001/mcp") as client:
        for step, step_i in zip(plan_steps, range(len(plan_steps))):

            step_action = step["action"].strip().lower()
            step_type = step["type"].strip().lower()
            step_name = step["name"].strip()

            print(f"Executing step {step_i}: {step_action}")

            if step_type == "resource":
                resource = resource_map[step_name]
                # Typically, resource should have .uri (or .name, but .uri is canonical)
                result = await client.read_resource(resource.uri)
                # Usually result is a list of ResponseItem, take 0th and .text or .json()
                value = (
                    result[0].text if hasattr(result[0], "text") else result[0]
                )
                memory[step_name] = value

            elif step_type == "tool":
                tool = tool_map[step_name]
                # Tools typically need named arguments as dict
                if step_name == "validate_simulation_parameters":
                    params = memory["parameters"]
                    result = await client.call_tool(
                        tool.name, {"parameters": params}
                    )
                    memory["validation"] = result.data
                elif step_name == "run_call_centre_simulation":
                    params = memory["parameters"]
                    result = await client.call_tool(
                        tool.name, {"parameters": params}
                    )
                    memory["simulation_result"] = result.data
                else:
                    result = await client.call_tool(tool.name)
                    memory[step_name] = result.data

            elif step_type == "prompt":

                prompt = prompt_map[step_name]
                # Most prompts need schema, user input
                schema = memory.get("get_experiment_parameter_schema")
                result = await client.get_prompt(
                    prompt.name, {"schema": schema, "user_input": user_input}
                )
                llm_prompt_text = result.messages[0].content.text
                # Actually run LLM (synchronously!)
                llm_result = llm.invoke(llm_prompt_text)
                parameters = json.loads(clean_llm_response(llm_result))
                memory["parameters"] = parameters

    return memory


async def main(
    planning_model_name: str = "gemma3:latest",
    summarising_model_name: str = "gemma3n:e4b",
    debug_mode: bool = False,
) -> None:
    """
    Main workflow function that orchestrates the entire simulation agent process.

    This function coordinates the complete workflow from user input to simulation results:
    1. Connects to the specified LLM model for planning and summarising...
    2. Fetches available prompts and lets LLM select the most suitable one
    3. Retrieves experiment schema and generates simulation parameters
    4. Executes the simulation and displays results

    Parameters
    ----------
    model_name : str, optional
        Name of the Ollama model to use for LLM reasoning (default: "gemma3:latest")

    Returns
    -------
    None
        Function prints results to console and returns None

    Raises
    ------
    ConnectionError
        If unable to connect to Ollama server or MCP server
    json.JSONDecodeError
        If LLM response cannot be parsed as JSON
    ValueError
        If no suitable prompt template is found

    Examples
    --------
    >>> asyncio.run(main("llama3:latest"))
    # Executes full workflow with llama3 model

    >>> asyncio.run(main("deepseek-r1:1.5b"))
    # Executes workflow with deepseek model
    """
    # Setup logging
    logger = setup_logging(debug_mode)

    if debug_mode:
        logger.debug(f"Planning model: {planning_model_name}")
        logger.debug(f"Summary model: {summarising_model_name}")

    console = Console()

    # Basic prompt with default
    user_input = Prompt.ask(
        "Simulation request:",
        default="e.g. Simulate 14 operators, 12 nurses, and 5% increase in demand.",
    )

    # 1. Connect to Ollama model
    llm = OllamaLLM(
        model=planning_model_name, base_url="http://localhost:11434"
    )

    # 2. Plan the task step by step
    planning_prompt = await create_task_planning_prompt(user_input)

    # store tools, resources and prompts for later...
    features = await fetch_all_features()

    if debug_mode:
        debug_print_available_features(features, logger)

    # Show progress indicator while LLM processes prompt selection
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]🧠 Planning modelling task..."),
        transient=True,
    ) as progress:
        task = progress.add_task("planning", total=None)
        response = llm.invoke(planning_prompt)
        progress.remove_task(task)

    # extract the plan from the LLM response
    plan_steps = parse_llm_plan(response)

    if debug_mode:
        logger.debug("=== LLM PLANNING RESPONSE ===")
        logger.debug(f"Response length: {len(response)} characters")
        logger.debug("Response preview:")
        logger.debug(
            response[:500] + "..." if len(response) > 500 else response
        )
        debug_print_plan(plan_steps, logger)

    # 3. Run the plan
    # Show progress indicator while plan is running
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]⚙️  Executing the plan..."),
        transient=True,  # Removes progress bar after completion
    ) as progress:
        task = progress.add_task("executing", total=None)
        memory = await run_plan(plan_steps, features, llm, user_input)
        progress.remove_task(task)

    # 4. Results
    # 2nd LLM for summarising the parameters - could probably do this with code faster maybe.
    llm = OllamaLLM(
        model=summarising_model_name, base_url="http://localhost:11434"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]🧠 Summarising parameters used..."),
        transient=True,  # Removes progress bar after completion
    ) as progress:
        task = progress.add_task("summarising", total=None)
        prompt = PromptTemplate.from_template(PARAMETER_TABLE_TEMPLATE)
        param_display_prompt = prompt.format(json_object=memory["parameters"])
        llm_result = llm.invoke(param_display_prompt)
        progress.remove_task(task)

    # display parameters
    console.print(Markdown(llm_result))

    # Display simulation results in a formatted table
    console.print(Markdown("✅ **Simulation complete.**"))
    df = pd.DataFrame(
        list(memory["simulation_result"].items()), columns=["KPIs", "Values"]
    )
    console.print(Markdown(df.round(2).to_markdown()))


def parse_arguments():
    """Parse command line arguments for model selection."""
    parser = argparse.ArgumentParser(
        description="Simulation Agent Workflow - Experiment with a simulation model using natural",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python agent_planning_workflow.py -p gemma3:27b -s gemma3:27b
            python agent_planning_workflow.py --planning deepseek-r1:32b --summary llama3.1:8b
            python agent_planning_workflow.py --debug  # Uses default models with debug enabled
            python agent_planning_workflow.py -p gemma3:27b -s llama3.1:8b -d  # Custom models with debug
            """,
    )

    parser.add_argument(
        "-p",
        "--planning",
        type=str,
        default="gemma3:27b",
        help="Model to use for reasoning and planning (default:gemma3:27b)",
    )

    parser.add_argument(
        "-s",
        "--summary",
        type=str,
        default="llama3:latest",
        help="Model to use for summarizing parameters (default: gemma3n:e4b)",
    )

    # Add the debug flag
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output and detailed logging",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Available model options for testing
    # model_name = "gemma3n:e4b"
    # model_name = "deepseek-r1:32b"
    # model_name = "llama3:latest"
    # model_name = "llama3.1:8b"
    # model_name = "gemma3:27b"
    # model_name = "gemma3:27b-it-qat"
    # model_name = "qwen2-math:7b"
    # model_name = "mistral:7b"

    # Parse command line arguments
    args = parse_arguments()

    # Run the main workflow with debug mode if specified
    asyncio.run(main(args.planning, args.summary, debug_mode=args.debug))
