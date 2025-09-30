
import streamlit as st
import asyncio
import json
import subprocess
import time
import threading
import queue
from io import StringIO
import sys
from contextlib import redirect_stdout, redirect_stderr
import os
from typing import Optional, Dict, Any
import pandas as pd

# Import the agents (assuming they're in the same directory or on path)
try:
    # These would be your actual imports
    from langchain_ollama import OllamaLLM
    from fastmcp import Client
    from agent_planning_workflow import main as planning_main
    from agent_self_reflection import main as reflection_main
    import agent_planning_workflow
    import agent_self_reflection
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all dependencies are installed and agents are accessible.")

# Page configuration
st.set_page_config(
    page_title="DES Agent Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .debug-output {
        background-color: #2e2e2e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        max-height: 400px;
        overflow-y: auto;
    }
    .simulation-results {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'execution_logs' not in st.session_state:
    st.session_state.execution_logs = []
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'mcp_server_status' not in st.session_state:
    st.session_state.mcp_server_status = "Unknown"


import requests

def get_ollama_models():
    """Get list of models from Ollama server"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data["models"]]
            return models
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []



# Header
st.markdown('<h1 class="main-header">🤖 DES Agent Interface</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Agent selection
agent_type = st.sidebar.selectbox(
    "Select Agent Type",
    ["Dynamic Planner", "Self-Reflective"],
    help="Choose between the two agent architectures"
)

# Model configuration based on agent type
st.sidebar.subheader("🧠 LLM Configuration")

# get the model list from Ollama and get default indexese
available_models = get_ollama_models()
try:
    gemma_index = available_models.index("gemma3:27b")
except ValueError:
    gemma_index = 0

try:
    llama_index = available_models.index("lamma3:latest")
except:
    llama_index = 0

if agent_type == "Dynamic Planner":
    planning_model = st.sidebar.selectbox(
        "Planning Model",
        available_models,
        index=gemma_index,  # Default to gemma3:27b
        help="Model used for reasoning and task planning"
    )

    summary_model = st.sidebar.selectbox(
        "Summary Model", 
        available_models,
        index=llama_index,  # Default to llama3:latest
        help="Model used for summarizing parameters and results"
    )

    planning_debug = st.sidebar.checkbox(
        "Enable Planning Debug Mode",
        help="Show detailed workflow and planning information"
    )

else:  # Self-Reflective Agent
    reflection_model = st.sidebar.selectbox(
        "Reflection Model",
        available_models,
        index=gemma_index,  # Default to gemma3:27b
        help="Model used for parameter generation and self-reflection"
    )

    reflection_debug = st.sidebar.checkbox(
        "Enable Reflection Debug Mode",
        help="Show detailed reflection and retry information"
    )

# Server status check
st.sidebar.subheader("🌐 Server Status")

def check_server_status(url: str, server_type: str = "generic", timeout: int = 3) -> str:
    """Check if a server is running with proper protocol support"""
    try:
        if server_type == "mcp":
            # MCP uses JSON-RPC - check if port is listening
            import socket
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            host = parsed.hostname or 'localhost'
            port = parsed.port or 8001
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            return "Running" if result == 0 else "Not Running"
                
        elif server_type == "ollama":
            response = requests.get(url, timeout=timeout)
            return "Running" if response.status_code == 200 else "Error"
            
    except Exception as e:
        return "Not Running"

# Server status indicators
mcp_status = mcp_status = check_server_status("http://localhost:8001/mcp", server_type="mcp") 
ollama_status = check_server_status("http://localhost:11434/api/tags", server_type="ollama")

st.sidebar.markdown(f"**MCP Server:** {'🟢' if mcp_status == 'Running' else '🔴'} {mcp_status}")
st.sidebar.markdown(f"**Ollama Server:** {'🟢' if ollama_status == 'Running' else '🔴'} {ollama_status}")

if mcp_status != "Running":
    st.sidebar.warning("⚠️ MCP Server not detected. Start with: `python mcp_server.py`")
if ollama_status != "Running":
    st.sidebar.warning("⚠️ Ollama Server not detected. Start with: `ollama serve`")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Simulation Request")

    # Pre-defined examples
    example_requests = [
        "Simulate 14 operators, 12 nurses and 5% extra demand",
        "Run scenario with high staffing and normal call volume", 
        "Test configuration with minimal staff (5 operators, 3 nurses)",
        "Simulate peak hours with 20 operators, 15 nurses and 20% increased demand",
        "Run baseline scenario with default parameters"
    ]

    selected_example = st.selectbox(
        "Quick Examples:",
        ["Custom Input"] + example_requests,
        help="Select a pre-defined example or choose 'Custom Input'"
    )

    if selected_example == "Custom Input":
        user_input = st.text_area(
            "Enter your simulation request:",
            placeholder="e.g., Simulate 14 operators, 12 nurses and 5% extra demand",
            height=100
        )
    else:
        user_input = st.text_area(
            "Simulation request:",
            value=selected_example,
            height=100
        )

with col2:
    st.subheader("🎯 Agent Information")

    if agent_type == "Dynamic Planner":
        st.markdown("""
        <div class="agent-card">
        <h4>🧠 Dynamic Planner Agent</h4>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Dual LLM architecture</li>
            <li>Dynamic task planning</li>
            <li>Memory-driven execution</li>
            <li>Step-by-step workflow</li>
        </ul>
        <p><strong>Models:</strong></p>
        <ul>
            <li>Planning: """ + planning_model + """</li>
            <li>Summary: """ + summary_model + """</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="agent-card">
        <h4>🔄 Self-Reflective Agent</h4>
        <p><strong>Features:</strong></p>
        <ul>
            <li>LangGraph state machine</li>
            <li>Validation-driven learning</li>
            <li>Bounded retry logic</li>
            <li>Error analysis & reflection</li>
        </ul>
        <p><strong>Model:</strong> """ + reflection_model + """</p>
        </div>
        """, unsafe_allow_html=True)

# Execution section
st.subheader("🚀 Execute Simulation")

# Create execution button
if st.button("Run Simulation", type="primary", disabled=(not user_input.strip())):
    if mcp_status != "Running" or ollama_status != "Running":
    #if ollama_status != "Running":
        st.error("❌ Please ensure both MCP and Ollama servers are running before executing.")
    else:
        with st.spinner("🤖 Agent is working..."):
            # Create placeholders for live updates
            status_placeholder = st.empty()
            debug_placeholder = st.empty()
            results_placeholder = st.empty()

            # Capture agent execution
            class StreamCapture:
                def __init__(self):
                    self.logs = []
                    self.current_status = "Initializing..."

                def write(self, text):
                    if text.strip():
                        self.logs.append(text.strip())
                        # Update status based on content
                        if "Planning modelling task" in text:
                            self.current_status = "🧠 Planning simulation task..."
                        elif "Executing the plan" in text:
                            self.current_status = "⚙️ Executing plan steps..."
                        elif "Reasoning about simulation parameters" in text:
                            self.current_status = "🤔 Generating parameters..."
                        elif "Summarising parameters" in text:
                            self.current_status = "📊 Summarizing results..."
                        elif "Simulation complete" in text:
                            self.current_status = "✅ Simulation complete!"

                def flush(self):
                    pass

            capture = StreamCapture()

            try:
                # Redirect stdout and stderr to capture agent output
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = capture
                sys.stderr = capture

                # Execute the selected agent
                if agent_type == "Dynamic Planner":
                    # Simulate the agent execution (in real implementation, you'd call the actual functions)
                    status_placeholder.info("🧠 Starting Dynamic Planner Agent...")

                    # In real implementation, you would do:
                    # result = asyncio.run(planning_main(planning_model, summary_model, planning_debug))

                    # For demonstration, simulate the process
                    time.sleep(1)
                    status_placeholder.info("📋 Generating execution plan...")

                    if planning_debug:
                        debug_placeholder.markdown("""
                        <div class="debug-output">
                        🐛 Debug mode enabled - showing workflow details<br>
                        === MCP SERVER CAPABILITIES ===<br>
                        🔧 Available tools (3): run_call_centre_simulation, validate_simulation_parameters<br>
                        📚 Available resources (2): schema/experiment_parameters, model/description<br>
                        📝 Available prompts (1): parameter_jsonification_prompt<br>
                        === LLM GENERATED PLAN ===<br>
                        Step 1: Get experiment parameter schema<br>
                        → Type: Resource, Name: get_experiment_parameter_schema<br>
                        Step 2: Generate simulation parameters<br>
                        → Type: Prompt, Name: parameter_jsonification_prompt<br>
                        Step 3: Validate parameters<br>
                        → Type: Tool, Name: validate_simulation_parameters<br>
                        Step 4: Run simulation<br>
                        → Type: Tool, Name: run_call_centre_simulation
                        </div>
                        """, unsafe_allow_html=True)

                    time.sleep(2)
                    status_placeholder.info("⚙️ Executing planned steps...")
                    time.sleep(2)
                    status_placeholder.success("✅ Simulation complete!")

                else:  # Self-Reflective Agent
                    status_placeholder.info("🔄 Starting Self-Reflective Agent...")

                    if reflection_debug:
                        debug_placeholder.markdown("""
                        <div class="debug-output">
                        🐛 Debug mode enabled - showing reflection details<br>
                        🧠 Reasoning about simulation parameters...<br>
                        ✅ Parameter validation successful<br>
                        📊 Parameters formatted successfully<br>
                        🏥 Running healthcare call centre simulation...
                        </div>
                        """, unsafe_allow_html=True)

                    time.sleep(1)
                    status_placeholder.info("🤔 Generating parameters...")
                    time.sleep(2)  
                    status_placeholder.info("✅ Parameters validated successfully")
                    time.sleep(1)
                    status_placeholder.info("🏥 Running simulation...")
                    time.sleep(2)
                    status_placeholder.success("✅ Simulation complete!")

                # Display mock results (in real implementation, these would come from the agents)
                sample_parameters = {
                    "n_operators": 14,
                    "n_nurses": 12, 
                    "mean_iat": 0.57,
                    "call_low": 5.0,
                    "call_mode": 7.0,
                    "call_high": 10.0,
                    "callback_prob": 0.4,
                    "run_length": 1000,
                    "random_seed": 42
                }

                sample_results = {
                    "01_mean_waiting_time": 2.45,
                    "02_operator_util": 78.5,
                    "03_mean_nurse_waiting_time": 1.23,
                    "04_nurse_util": 65.2,
                    "05_callback_rate": 39.8
                }

                # Display results
                results_placeholder.markdown("""
                <div class="simulation-results">
                <h3>🎯 Simulation Results</h3>
                </div>
                """, unsafe_allow_html=True)

                # Parameters table
                st.subheader("📊 Parameters Used")
                param_df = pd.DataFrame(list(sample_parameters.items()), columns=['Parameter', 'Value'])
                st.dataframe(param_df, use_container_width=True)

                # Results table  
                st.subheader("📈 Key Performance Indicators")
                results_df = pd.DataFrame(list(sample_results.items()), columns=['KPI', 'Value'])
                st.dataframe(results_df, use_container_width=True)

                # Store results in session state
                st.session_state.simulation_results = {
                    'parameters': sample_parameters,
                    'results': sample_results,
                    'agent_type': agent_type,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }

            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")

            finally:
                # Restore stdout and stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

# Results history section
if st.session_state.simulation_results:
    st.subheader("📚 Recent Results")

    with st.expander("View Last Simulation Results", expanded=False):
        results = st.session_state.simulation_results

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Agent Type:**", results['agent_type'])
            st.write("**Timestamp:**", results['timestamp'])

        with col2:
            if st.button("📋 Copy Parameters JSON"):
                st.code(json.dumps(results['parameters'], indent=2))

        st.write("**Parameters:**")
        st.json(results['parameters'])

        st.write("**Results:**")  
        st.json(results['results'])

# Footer with helpful information
# st.markdown("---")
# st.markdown("""
# ### 💡 Tips for Better Results

# **For Dynamic Planner:**
# - Use gemma3:27b for best planning performance
# - Enable debug mode to see step-by-step execution
# - Try complex requests that require multi-step reasoning

# **For Self-Reflective Agent:**
# - gemma3:27b and mistral:7b work well for parameter generation
# - The agent will automatically retry if validation fails
# - Watch for reflection patterns in debug mode

# **Common Request Patterns:**
# - "Simulate X operators, Y nurses and Z% extra demand"
# - "Run scenario with [adjective] staffing and [adjective] call volume"
# - "Test configuration with [specific parameters]"
# """)

# # Troubleshooting section
# with st.expander("🔧 Troubleshooting", expanded=False):
#     st.markdown("""
#     **Common Issues:**

#     1. **MCP Server Not Running:**
#        ```bash
#        python mcp_server.py
#        ```
#        Server should be available at http://localhost:8001/mcp

#     2. **Ollama Server Not Running:**
#        ```bash
#        ollama serve
#        ```
#        Then pull required models:
#        ```bash
#        ollama pull gemma3:27b
#        ollama pull llama3:latest
#        ollama pull mistral:7b
#        ```

#     3. **Model Not Found:**
#        Check available models: `ollama list`

#     4. **Connection Errors:**
#        - Check firewall settings
#        - Ensure ports 8001 and 11434 are available
#        - Verify model names match exactly
#     """)