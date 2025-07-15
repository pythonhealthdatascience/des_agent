from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from fastmcp import Client
import asyncio
import json

async def fetch_schema():
    """Fetch experiment schema using FastMCP client"""
    async with Client("http://localhost:8000/mcp") as client:
        resources = await client.list_resources()
        for resource in resources:
            print(f"Resource URI: {resource.uri}")
            print(f"Name: {resource.name}")
            print(f"Description: {resource.description}")
            print(f"MIME Type: {resource.mimeType}")
        result = await client.read_resource("resource://get_experiment_template")
        print(result)
        print(type(result))
        return json.loads(result)

async def run_simulation(parameters):
    """Run simulation using FastMCP client"""
    async with Client("http://localhost:8000/mcp") as client:
        result = await client.call_tool("run_experiment", parameters)
        return result.data

async def main():
    # 1. Connect to Ollama model
    llm = OllamaLLM(model="gemma3:latest", base_url="http://localhost:11434")
    
    # 2. Fetch experiment schema using FastMCP
    schema = await fetch_schema()
    print("Schema retrieved:", schema)
    
    # 3. Create LangChain prompt
    PROMPT_TEMPLATE = """
    You are a discrete-event simulation assistant. 
    
    You help turn natural language into structured simulation parameters.
    
    Here is the simulation configuration schema:
    {schema}
    
    User request:
    {user_input}
    
    Return the parameters as a JSON object.
    """
    
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    
    # 4. Example usage
    user_input = "Run with 14 operators and 12 nurses"
    response = chain.invoke({"schema": schema, "user_input": user_input})
    
    # 5. Parse and run simulation
    parameters = json.loads(response)
    result = await run_simulation(parameters)
    print("Simulation result:", result)

if __name__ == "__main__":
    asyncio.run(main())
