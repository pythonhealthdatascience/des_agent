"""
Simulation Agent Workflow Module

This module provides an intelligent agent workflow for configuring and running simulation models.
The agent uses LLM (Large Language Model) reasoning to interpret user requests, select appropriate
prompt templates, and generate simulation parameters automatically.

Key Features:
- Automatic prompt selection based on user input
- LLM-based parameter generation for simulation models
- FastMCP client integration for server communication
- Rich progress indicators for user feedback
- JSON response cleaning and validation

The workflow follows these steps:
1. User provides a natural language request
2. Agent selects the most suitable prompt template
3. LLM generates simulation parameters based on the request
4. Simulation is executed with the generated parameters
5. Results are formatted and displayed to the user

Example:
    Run the simulation with custom staffing levels:
    
    >>> asyncio.run(main("llama3:latest"))
    # User input: "Run with 14 operators and 12 nurses"
    # Agent automatically configures and runs the simulation
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

# prompt for terminal - nothing to do with LLM
from rich.prompt import Prompt

# Prompt template for LLM-based prompt selection
SELECTION_PROMPT_TEMPLATE = """
You are an assistant that helps select the most suitable prompt template for a user's request from a given list.

Available prompt options:
{prompt_options}

User instruction:
"{user_input}"

Your task:
- Carefully read the user instruction and the available prompt options.
- Select the prompt that best fits the user's instruction.
- Output ONLY the name of the most suitable prompt
- If none are suitable, respond with "None".

Output format:
prompt_name: <name>
reason: <reason>
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
    "    - If a step requires information not yet obtained (such as parameter names/types), plan to retrieve it first.\n"
    "    - Briefly explain the purpose of each step.\n\n"
    "Output Format:\n"
    "Return your plan as a numbered list. For each step, use the following structure:\n\n"
    "Step N:\n"
    "- Action: [Clear description of what you will do in this step]\n"
    "- Type: [Resource/Tool/Prompt]\n"
    "- Name: [Exact name from available list]\n"
    "- Rationale: [Brief explanation of why this step is necessary]\n"
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
        line = f"{i}. {p.name}: {p.description}" if not short else f"{p.name}: {p.description}"
        formatted.append(line)
    return "\n".join(formatted)


async def fetch_prompts() -> str:
    """
    Fetch all available prompts from the MCP server.
    
    Connects to the FastMCP server and retrieves all available prompt templates.
    The prompts are then formatted into a numbered list suitable for LLM processing.
    
    Returns
    -------
    str
        Formatted string containing all available prompts with descriptions
        
    Raises
    ------
    ConnectionError
        If unable to connect to the MCP server at localhost:8001
    """
    async with Client("http://localhost:8001/mcp") as client:
        prompts = await client.list_prompts()
        return format_prompt_options(prompts)


async def fetch_all_features() -> dict:
    """Fetch all tools, resources, and prompts from the MCP server asynchronously."""
    async with Client("http://localhost:8001/mcp") as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
    return {"tools": tools, "resources": resources, "prompts": prompts}


async def create_task_planning_prompt(user_input: str) -> str:
    """
    Assemble a step-planning prompt for the agent, embedding available tools, resources, and prompts.

    This function asynchronously queries the MCP server for the current set of tools, resources,
    and prompt templates, then formats a human- and LLM-readable planning prompt using a 
    globally-defined template string (e.g., TASK_PLANNING_PROMPT_TEMPLATE). The planning prompt 
    is designed to instruct the agent on how to break down a user request into stepwise actions.

    Parameters
    ----------
    user_input : str
        The user's natural language request or instruction for running or configuring a simulation.

    Returns
    -------
    formatted_prompt : str
        The agent planning prompt containing lists of available MCP capabilities and the user query,
        ready to be sent to an LLM or agent step planner.

    Examples
    --------
    >>> user_input = "Run with 10 operators and double the demand."
    >>> prompt = await create_task_planning_prompt(user_input)
    >>> print(prompt)
    (prints out the planning prompt with tools, resources, prompts, and user input)
    """
    features = await fetch_all_features()
    tool_names = [getattr(t, "name", str(t)) for t in features.get("tools", [])]
    resource_names = [getattr(r, "name", str(r)) for r in features.get("resources", [])]
    prompt_names = [getattr(p, "name", str(p)) for p in features.get("prompts", [])]

    prompt = PromptTemplate.from_template(TASK_PLANNING_PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(
        tools="\n".join(tool_names) if tool_names else "None",
        resources="\n".join(resource_names) if resource_names else "None",
        prompts="\n".join(prompt_names) if prompt_names else "None",
        user_input=user_input
    )
    return formatted_prompt



async def fetch_schema() -> Dict[str, Any]:
    """
    Fetch experiment schema from the MCP server.
    
    Retrieves the experiment template schema that defines the structure
    and parameters available for simulation configuration.
    
    Returns
    -------
    Dict[str, Any]
        JSON schema dictionary containing experiment parameter definitions
        
    Raises
    ------
    ConnectionError
        If unable to connect to the MCP server
    json.JSONDecodeError
        If the returned schema is not valid JSON
    """
    async with Client("http://localhost:8001/mcp") as client:
        result = await client.read_resource("resource://get_experiment_template")
        return json.loads(result[0].text)


async def run_simulation(parameters: Dict[str, Any]) -> Any:
    """
    Execute simulation with the provided parameters.
    
    Sends simulation parameters to the MCP server and executes the experiment.
    The server runs the simulation model and returns the results.
    
    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary containing simulation parameters (e.g., staffing levels,
        operational settings, time horizons)
    
    Returns
    -------
    Any
        Simulation results data structure containing KPIs and metrics
        
    Raises
    ------
    ConnectionError
        If unable to connect to the MCP server
    ValueError
        If parameters are invalid or incomplete
    """
    async with Client("http://localhost:8001/mcp") as client:
        result = await client.call_tool("run_experiment", {"parameters": parameters})
        return result.data




async def validate_params(parameters: Dict[str, Any]) -> (bool, list):
    """
    Use the MCP server to check parameters.
    Returns (is_valid, errors).
    """
    async with Client("http://localhost:8001/mcp") as client:
        result = await client.call_tool("validate_parameters", {"parameters": parameters})
        # result.data should be your dict: {"is_valid": bool, "errors": [str, ...]}
        return result.data.get("is_valid", False), result.data.get("errors", [])



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
        response = response[3:]   # Remove ```
    if response.endswith("```"):
        response = response[:-3]  # Remove closing ```
    
    # Find JSON object boundaries
    response = response.strip()
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        return response[start_idx:end_idx+1]
    
    return response


async def main(model_name: str = "gemma3:latest") -> None:
    """
    Main workflow function that orchestrates the entire simulation agent process.
    
    This function coordinates the complete workflow from user input to simulation results:
    1. Connects to the specified LLM model
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
    # Example user input - in practice, this would come from user interface
    # user_input = "Run with 14 operators and 12 nurses"


    # Basic prompt with default
    user_input = Prompt.ask(
        "Simulation request:", 
        default="e.g. Run with 14 operators, 12 nurses, and 5% increase in demand.")
    
    # 1. Connect to Ollama model
    llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")

    # 2. Fetch available prompts from MCP server
    prompts = await fetch_prompts()
    
    # 3. Plan the task step by step
    planning_prompt = await create_task_planning_prompt(user_input)

    # Show progress indicator while LLM processes prompt selection
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]🧠 Planning modelling task..."),
        transient=True, 
    ) as progress:
        task = progress.add_task("reviewing", total=None)
        response = llm.invoke(planning_prompt)
        progress.remove_task(task)  

    print(response)

    # # 3. Use LLM to select the most appropriate prompt for the user's task
    # selection_prompt = PromptTemplate.from_template(SELECTION_PROMPT_TEMPLATE)
    
    # # Fill the prompt selection template with available options and user input
    # selection_input = selection_prompt.format(
    #     prompt_options=prompts,
    #     user_input=user_input 
    # )

    # # Show progress indicator while LLM processes prompt selection
    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[bold green]Reviewing available actions..."),
    #     transient=True,  # Removes progress bar after completion
    # ) as progress:
    #     task = progress.add_task("reviewing", total=None)
    #     response = llm.invoke(selection_input)
    #     progress.remove_task(task)  

    # print(f"Decision:\n {response}\n")

    # # Extract the selected prompt name using regex matching
    # match = re.search(r"prompt_name:\s*(\w+)", response)
    # if match:
    #     selected_prompt_name = match.group(1)
    # else:
    #     # No suitable prompt template found
    #     print("I cannot help with this task. Please enter a different query")
    #     return
    
    # # 4. Fetch experiment schema from MCP server
    # schema = await fetch_schema()
    # print("Simulation schema retrieved")

    # # 5. Retrieve the selected prompt template and populate with schema and user input
    # async with Client("http://localhost:8001/mcp") as client:
    #     chosen_prompt = await client.get_prompt(selected_prompt_name, {
    #         "schema": schema,
    #         "user_input": user_input
    #     })

    # # Extract the prompt text from the response
    # prompt_message = chosen_prompt.messages[0]
    # prompt_text = prompt_message.content.text
    
    # # Create the LLM chain (direct invocation without additional templates)
    # chain = llm
    
    # # Display the final prompt being sent to the LLM
    # # print("\n🧠 Final prompt to LLM:\n", prompt_text)
    
    # # Show progress indicator while LLM generates parameters
    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[bold blue]Thinking about model parameters 🧠 ..."),
    #     transient=True,  # Removes progress bar after completion
    # ) as progress:
    #     task = progress.add_task("thinking", total=None)
    #     response = chain.invoke(prompt_text)
    #     progress.remove_task(task)

    # print("Chosen parameters")
    # # Clean the LLM response to remove markdown formatting
    # cleaned_response = clean_llm_response(response)
    # print(cleaned_response)

    # # 7. Parse parameters and execute simulation
    # parameters = json.loads(cleaned_response)



    # # Show progress indicator during simulation execution
    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[bold green]Validating parameters and simulating..."),
    #     transient=True, 
    # ) as progress:
    #     task = progress.add_task("simulating", total=None)

    #     # validate parameters
  
    #     is_valid, errors = await validate_params(parameters)
    #     if not is_valid:
    #         print("Parameter errors found. Terminating early.")
    #         for err in errors:
    #             print("-", err)
    #         return
    #     else:
    #         # Proceed with simulation
    #         result = await run_simulation(parameters)

    #     progress.remove_task(task) 
    
    # # Display simulation results in a formatted table
    # print("Simulation result:")
    # df = pd.DataFrame(result.items(), columns=['KPIs', 'Values']).round(2)
    # print(df)


if __name__ == "__main__":
    # Available model options for testing
    # model_name = "gemma3n:e2b"
    #model_name = "deepseek-r1:8b"
    #model_name = "llama3:latest"
    #model_name = "llama3.1:8b"
    model_name = "gemma3:27b"
    #model_name = "qwen2-math:7b"
    #model_name = "mistral:7b"
    # Run the main workflow
    asyncio.run(main(model_name))
