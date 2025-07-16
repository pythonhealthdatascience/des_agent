from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from fastmcp import Client

import asyncio
import json
import pandas as pd
import re

from rich.progress import Progress, SpinnerColumn, TextColumn

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

def format_prompt_options(prompt_list, short=False):
    """Convert MCP prompts into numbered or named options for LLM display."""
    formatted = []
    for i, p in enumerate(prompt_list, 1):
        line = f"{i}. {p.name}: {p.description}" if not short else f"{p.name}: {p.description}"
        formatted.append(line)
    return "\n".join(formatted)

async def fetch_prompts():
    """Fetch all prompts from the server"""
    async with Client("http://localhost:8001/mcp") as client:
        prompts = await client.list_prompts()
        # Generate the numbered list for the selection message
        return format_prompt_options(prompts)

async def fetch_schema():
    """Fetch experiment schema using FastMCP client"""
    async with Client("http://localhost:8001/mcp") as client:
        result = await client.read_resource("resource://get_experiment_template")
        return json.loads(result[0].text)

async def run_simulation(parameters):
    """Run simulation using FastMCP client"""
    async with Client("http://localhost:8001/mcp") as client:
        result = await client.call_tool("run_experiment",  
                                        {"parameters": parameters})
        return result.data

def clean_llm_response(response):
    """Clean LLM response to extract JSON from markdown blocks"""
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


async def main(model_name="gemma3:latest"):

    # 0. Example usage
    user_input = "Run with 14 operators and 12 nurses"
    
    # 1. Connect to Ollama model
    llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")

    # 2. List available prompts
    prompts = await fetch_prompts()
    
    # 3. Prompt LLM to choose the prompt to use for the users task.   
    selection_prompt = PromptTemplate.from_template(SELECTION_PROMPT_TEMPLATE)
    
    # Fill the prompt selection template
    selection_input = selection_prompt.format(
        prompt_options=prompts,
        user_input=user_input 
    )

    # show progress bar...
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Reviewing available actions..."),
        transient=True,  # Removes progress bar after completion
    ) as progress:
        task = progress.add_task("reviewing", total=None)
        response = llm.invoke(selection_input)
        progress.remove_task(task)  

    print(f"Decision:\n {response}\n")

    match = re.search(r"prompt_name:\s*(\w+)", response)
    if match:
        selected_prompt_name = match.group(1)
    else:
        # no suitable prompt templated.
        print("I cannot help with this task. Please enter a different query")
        return
    
    # 4. Fetch experiment schema using FastMCP
    schema = await fetch_schema()
    print("Simulation schema retrieved")

    # 5. Retrieve the prompt template and populate with data
    async with Client("http://localhost:8001/mcp") as client:
        chosen_prompt = await client.get_prompt(selected_prompt_name, {
            "schema": schema,
            "user_input": user_input
        })

    # At this point, chosen_prompt.content.text is a final ready-to-go message (string)
    prompt_message = chosen_prompt.messages[0]
    prompt_text = prompt_message.content.text  # Access the text correctly
    
    # Create the chain: just send the filled prompt as user input
    chain = llm  # No extra prompt template needed
    
    # Optional: log what's about to be sent to the LLM
    print("\n🧠 Final prompt to LLM:\n", prompt_text)
    
    # show progress bar...
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Thinking about model parameters..."),
        transient=True,  # Removes progress bar after completion
    ) as progress:
        task = progress.add_task("thinking", total=None)
        response = chain.invoke(prompt_text)
        progress.remove_task(task)  # Optional, as exiting the context manager stops it

    print("Chosen parameters")
    # Clean the response to remove markdown if present
    cleaned_response = clean_llm_response(response)
    print(cleaned_response)

    # 7. Parse and run simulation
    parameters = json.loads(cleaned_response)

    # show progress bar...
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Simulating..."),
        transient=True, 
    ) as progress:
        task = progress.add_task("simulating", total=None)
        result = await run_simulation(parameters)
        progress.remove_task(task) 
    
    print("Simulation result:")
    df = pd.DataFrame(result.items(), columns=['KPIs', 'Values']).round(2)
    print(df)

if __name__ == "__main__":
    #model_name = "gemma3n:e2b"
    #model_name = "deepseek-r1:1.5b"
    model_name = "llama3:latest"
    asyncio.run(main(model_name))
