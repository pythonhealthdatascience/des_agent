"""
Script to add expected simulation results to the evals dataset
single rep at this stage.
"""
import json
from mcpsim.example_models.callcentre.call_centre_model import run_simulation_from_dict 

# Load test cases from evals_input.json
with open("evals/evals_input.json", "r") as f:
    scenarios = json.load(f)

# Run each scenario once and attach results
for key, obj in scenarios.items():
    params = obj["parameters"]
    results = run_simulation_from_dict(params)
    obj["expected_results"] = results

# Save updated scenarios with results to evals.json
with open("evals/evals.json", "w") as f:
    json.dump(scenarios, f, indent=2)

print("Simulation results saved to evals.json")
