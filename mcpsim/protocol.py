"""Defines a simple protocol that each simulation model should 
use in order to be compatable with the agent.
"""

from typing import Protocol, runtime_checkable, Dict, Any
from abc import abstractmethod

@runtime_checkable
class SimulationModelAdapter(Protocol):
    """Protocol defining the interface for MCP-compatible simulation models."""
    
    @abstractmethod
    def run_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the simulation with given parameters and return results."""
        ...
    
    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Return JSON schema for valid simulation parameters."""
        ...
    
    @abstractmethod
    def get_model_description(self) -> str:
        """Return human-readable description of the simulation model."""
        ...
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters and return validation results."""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier for this simulation model."""
        ...