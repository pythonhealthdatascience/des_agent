from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from fastmcp import Client
import asyncio
import json
import pandas as pd

from rich.progress import Progress, SpinnerColumn, TextColumn

async def fetch_schema():
    """Fetch experiment schema using FastMCP client"""
    async with Client("http://localhost:8001/mcp") as client:
        # resources = await client.list_resources()
        # for resource in resources:
        #     print(f"Resource URI: {resource.uri}")
        #     print(f"Name: {resource.name}")
        #     print(f"Description: {resource.description}")
        #     print(f"MIME Type: {resource.mimeType}")
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
    # 1. Connect to Ollama model
    llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
    
    # 2. Fetch experiment schema using FastMCP
    schema = await fetch_schema()
    print("Simulation schema retrieved")
    
    # 3. Create LangChain prompt
    PROMPT_TEMPLATE = """
    You are a discrete-event simulation assistant.
    You help turn natural language into structured simulation parameters.
    
    Here is the simulation configuration schema:
    {schema}
    
    User request: {user_input}
    
    IMPORTANT: 
    - Return ONLY the parameter VALUES, not the schema structure
    - Use the exact parameter names from the schema
    - Output a simple JSON object with key-value pairs
    - Do NOT include type information, descriptions, or constraints
    
    Example output format:
    {{
      "n_operators": 15,
      "n_nurses": 8,
      "mean_iat": 0.5,
      "random_seed": 42,
      "run_length": 1000
    }}
    
    JSON Response:
    """

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    
    # 4. Example usage
    user_input = "Run with 14 operators and 12 nurses"
    # print("Generating simulation parameters...")
    # response = chain.invoke({"schema": schema, "user_input": user_input})

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Thinking about model parameters..."),
        transient=True,  # Removes progress bar after completion
    ) as progress:
        task = progress.add_task("thinking", total=None)
        response = chain.invoke({"schema": schema, "user_input": user_input})
        progress.remove_task(task)  # Optional, as exiting the context manager stops it

    print("LLM response")
    # Clean the response to remove markdown
    cleaned_response = clean_llm_response(response)
    print(f"Cleaned response: {cleaned_response}")

    # 5. Parse and run simulation
    parameters = json.loads(cleaned_response)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Simulating..."),
        transient=True,  # Removes progress bar after completion
    ) as progress:
        task = progress.add_task("simulating", total=None)
        result = await run_simulation(parameters)
        progress.remove_task(task)  # Optional, as exiting the context manager stops it
    
    print("Simulation result:")
    df = pd.DataFrame(result.items(), columns=['KPIs', 'Values']).round(2)
    print(df)

if __name__ == "__main__":
    #model_name = "gemma3n:e2b"
    #model_name = "deepseek-r1:1.5b"
    model_name = "llama3:latest"
    asyncio.run(main(model_name))
