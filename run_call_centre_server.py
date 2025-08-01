from mcpsim.example_models import CallCentreSimulationAdapter
from mcpsim.server import SimulationMCPServer

def main():
    # Create the simulation model
    call_centre_sim = CallCentreSimulationAdapter()
    
    # Create the MCP server
    server = SimulationMCPServer(call_centre_sim)
    
    # Run the server
    server.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")

if __name__ == "__main__":
    main()