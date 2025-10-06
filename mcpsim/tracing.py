"""
Utility module to support tracing of the agent and tools using arize-phoenix
"""

import os
from phoenix.otel import register

def init_tracing(project_name: str = "sim-agent", endpoint: str = "http://localhost:6006"):
    """
    Initialise tracing

    Parameters:
    ----------
    project_name: str. optional (default="sim-agent")
        Name of project

    endpoint: str, optional (default = "http://localhost:6006")
        Port for Phoenix eval server.

    Returns:
        tracer

    """
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint
    return register(project_name=project_name, auto_instrument=True)
