import re

def parse_llm_plan(plan_text: str):
    """
    Converts a step-by-step plan output from an LLM into a list of dicts.
    """
    # Split by step (supports "Step N" or "Step N:")
    steps = re.split(r'\s*Step \d+:', plan_text)
    steps = [s.strip() for s in steps if s.strip()]   # Remove empties

    plan = []
    item_regex = {
        'action': r"- Action:\s*(.*)",
        'type': r"- Type:\s*(.*)",
        'name': r"- Name:\s*(.*)",
        'rationale': r"- Rationale:\s*(.*)",
    }

    for step in steps:
        entry = {}
        for key, regex in item_regex.items():
            match = re.search(regex, step, flags=re.IGNORECASE)
            if match:
                entry[key] = match.group(1).strip()
        if entry:
            plan.append(entry)
    return plan

# example
plan_text = """
Step 1: Retrieve the simulation parameter schema.
- Action: Obtain the schema that defines the input parameters for the healthcare call center simulation.
- Type: resource
- Name: get_experiment_parameter_schema
- Rationale: To understand the valid parameters and their types that can be used in the simulation.

Step 2: Validate the simulation parameters.
- Action: Validate the user's request against the retrieved schema to ensure the parameters are valid.
- Type: tool
- Name: validate_simulation_parameters
- Rationale: To ensure the parameters provided by the user are within the allowed range and format.

...
"""

parsed_steps = parse_llm_plan(plan_text)
from pprint import pprint
pprint(parsed_steps)
