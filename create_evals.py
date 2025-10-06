"""
Script to add expected simulation results to the evals dataset
single rep at this stage.
"""
import pandas as pd
import phoenix as px

import json
from datetime import datetime

# sim-agent imports
from mcpsim.example_models.callcentre.call_centre_model import run_simulation_from_dict 
from mcpsim.tracing import init_tracing

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

# ------------- upload evals to phoenix --------------------------
# Initialize a global tracer 
tracer_provider = init_tracing(project_name="mcp-agent-evaluation", endpoint="http://localhost:6006")

# convert to dataframe and transpose 
df_evals = pd.DataFrame(scenarios).T

# create a dataset consisting of input questions and expected outputs
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
phoenix_client = px.Client()
dataset = phoenix_client.upload_dataset(
    dataframe=df_evals, 
    dataset_name=f"initial_evals-{now}", 
    input_keys=["user_input"], 
    output_keys=["parameters", "expected_results"]
)

print("Simulation results saved to evals.json")
