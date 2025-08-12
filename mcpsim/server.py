import os

from fastmcp import FastMCP 
from langchain_core.prompts import PromptTemplate
from fastmcp.prompts.prompt import PromptMessage, TextContent

from mcpsim.protocol import SimulationModelAdapter

PARAMETER_PROMPT_PATH = "resources/parameter_prompt.txt"

from mcpsim.tracing import init_tracing  # your helper module from previous discussion

tracer_provider = init_tracing(project_name="sim-agent-evaluation")
tracer = tracer_provider.get_tracer("mcp-server-tracer")

class SimulationMCPServer:
    """MCP server that can work with any SimulationModelAdapter implementation."""
    
    def __init__(self, simulation_model: SimulationModelAdapter, server_name: str = None):
        # validate interface
        if not isinstance(simulation_model, SimulationModelAdapter):
            missing_methods = []
            for method in ['run_simulation', 'get_parameter_schema', 'get_model_description', 
                          'validate_parameters']:
                if not hasattr(simulation_model, method):
                    missing_methods.append(method)
            if not hasattr(simulation_model, 'model_name'):
                missing_methods.append('model_name (property)')
                
            raise TypeError(f"Object {type(simulation_model)} missing required methods: {missing_methods}")
        
        self.model = simulation_model
        server_name = server_name or f"{simulation_model.model_name} Simulation MCP Server"
        self.mcp = FastMCP(server_name)
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        """Register MCP tools that delegate to the simulation model."""
        
        # note investigate how to clean up description
        @tracer.tool(name="MCP.run_call_centre_simulation")
        @self.mcp.tool(
            name=f"run_{self.model.model_name}_simulation",
            description=f"""
            Runs a discrete-event {self.model.model_name} simulation with specified parameters, 
            returning performance metrics.

            Inputs: parameters (dict) — JSON object matching the experiment schema.
            Returns: dict with simulation metrics, such as mean wait times and resource utilizations.

            Tags: ["simulation", "{self.model.model_name}", "experiment"]
            """
        )
        def run_call_centre_simulation(parameters: dict) -> dict:
            return self.model.run_simulation(parameters)

        @tracer.tool(name="MCP.validate_simulation_parameters")
        @self.mcp.tool(
            name="validate_simulation_parameters",
            description="""
            Validate a proposed set of simulation parameters (JSON object) 
            against the experiment schema.

            Inputs: parameters (dict)
            Returns: {"is_valid": bool, "errors": [str, ...]} — status and explanation.

            Tags: ["validation", "parameter_check", "pre_run_check", "schema"]
            """
        )
        def validate_parameters(parameters: dict) -> dict:
            return self.model.validate_parameters(parameters)
    
    def _register_resources(self):
        """Register MCP resources that delegate to the simulation model."""
        
        @self.mcp.resource(
            uri="resource://schema/experiment_parameters",
            description="""
            Returns the JSON schema defining all allowed input parameters, parameter types, 
            and value constraints.

            Outputs: dict (JSON schema), sent as a JSON object.

            Tags: ["schema", "parameters", "template"]
            """
        )
        def get_schema() -> dict:
            return self.model.get_parameter_schema()

        @self.mcp.resource(
            uri="resource://model/description",
            description=f"""
            Provides a natural language description of 
            the {self.model.model_name} simulation model.

            Outputs: str (text description).

            Tags: ["model", "description", "documentation"]
            """
        )
        def get_description() -> str:
            return self.model.get_model_description()
    
    def _register_prompts(self):
        """Register MCP prompts for parameter conversion."""
        
        @self.mcp.prompt(
            name="parameter_jsonification_prompt",
            description="""
            INSTRUCTION TO LLM: Convert a user's freeform simulation request 
            into a JSON object matching a given schema.

            Inputs:
            - schema (str): JSON Schema as a string
            - user_input (str): User's natural language request

            Returns: PromptMessage (LLM input) guiding the agent to produce valid JSON parameters.

            Tags: ["jsonification", "schema_mapping", "prompt", "parameters"]
            """
        )
        def parameter_jsonification_prompt(
            schema: str, 
            user_input: str,
            validation_errors: str = ""
        ) -> PromptMessage:
            
            # handle path to schema file
            dir_path = os.path.dirname(os.path.realpath(__file__))
            parameter_prompt_path = os.path.join(dir_path, PARAMETER_PROMPT_PATH)

            with open(parameter_prompt_path, encoding="utf-8") as f:
                prompt_template_text = f.read()
            prompt = PromptTemplate.from_template(prompt_template_text)

            # Handle validation error feedback
            if validation_errors and validation_errors.strip():
                validation_feedback = (
                    "**Validation Feedback:**\n"
                    "Your last attempt did not pass validation for these reasons:\n"
                    f"{validation_errors}\n\n"
                    "Please address the issues above and try again."
                )
            else:
                validation_feedback = ""

            filled_prompt = prompt.format(
                schema=schema, 
                user_input=user_input,
                validation_feedback=validation_feedback
            )
            return PromptMessage(
                role="user",
                content=TextContent(type="text", text=filled_prompt)
            )
    
    def run(self, **kwargs):
        """Start the MCP server."""
        self.mcp.run(**kwargs)