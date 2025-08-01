import json
from model import run_simulation_from_dict
from typing import Dict, Any


class CallCentreSimulationAdapter:
    """Call centre simulation implementation following 
    the SimulationModelAdapter protocol."""
    
    def __init__(self, schema_path: str = "resources/schema.json"):
        self.schema_path = schema_path
        self._schema = None
    
    @property
    def model_name(self) -> str:
        return "urgent_care_call_centre"
    
    def run_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the discrete-event healthcare call centre simulation."""
        return run_simulation_from_dict(parameters)
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Load and return the parameter schema."""
        if self._schema is None:
            with open(self.schema_path) as f:
                self._schema = json.load(f)
        return self._schema
    
    def get_model_description(self) -> str:
        """Return human-readable description of the call centre model."""
        return (
            "This is a discrete-event simulation of a healthcare call centre. "
            "Patients call in, interact with operators, and a subset may require a nurse callback. "
            "Simulation components: SimPy queues and resources. Tracks wait times, utilization, and callback rates. "
            "Configurable parameters: number of operators and nurses, call durations and rates, etc. "
            "Sample: 'Run with 14 operators and 5% higher demand.'"
        )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate simulation parameters against the experiment schema."""
        schema = self.get_parameter_schema()
        errors = []
        
        for key, value in parameters.items():
            # Check for unknown parameters
            if key not in schema:
                errors.append(f"Unknown parameter: {key}")
                continue
                
            spec = schema[key]
            expected_type = int if spec["type"] == "int" else float
            
            # Type validation
            if not isinstance(value, expected_type):
                errors.append(f"{key} must be {spec['type']}")
                continue
                
            # Range validation
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"{key} below minimum {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"{key} above maximum {spec['maximum']}")
        
        # Cross-parameter validation
        if all(x in parameters for x in ("call_low", "call_mode", "call_high")):
            if not (parameters["call_low"] <= parameters["call_mode"] <= parameters["call_high"]):
                errors.append("call_low ≤ call_mode ≤ call_high violated")
                
        if all(x in parameters for x in ("nurse_consult_low", "nurse_consult_high")):
            if not (parameters["nurse_consult_low"] <= parameters["nurse_consult_high"]):
                errors.append("nurse_consult_low ≤ nurse_consult_high violated")
        
        return {"is_valid": len(errors) == 0, "errors": errors}