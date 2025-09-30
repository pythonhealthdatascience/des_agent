"""
This code was written by perplexity.ai labs initially. I've been editing it.
Likely to modify a lot as this isn't quite what I wanted.

"""
import streamlit as st
import asyncio
import json
import time
import sys
import os
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import requests
import socket
from urllib.parse import urlparse

# Import the agents
try:
    from langchain_ollama import OllamaLLM
    from fastmcp import Client
    from agent_planning_workflow import main as planning_main
    from agent_self_reflection import main as reflection_main
    import agent_planning_workflow
    import agent_self_reflection
    AGENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all dependencies are installed and agents are accessible.")
    AGENTS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sim Experimentation Agent Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'agent_busy' not in st.session_state:
    st.session_state.agent_busy = False


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

# maybe refactor into two seperate functions
def check_server_status(url: str, server_type: str = "generic", timeout: int = 3) -> str:
    """Check if a server (Ollama or Simulation MCP) is running with proper protocol support"""
    try:
        if server_type == "mcp":

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

async def run_self_reflection_agent(user_input: str, model_name: str, debug_mode: bool = False) -> Dict[str, Any]:
    """Run the actual self-reflection agent"""
    try:
        # Create the LLM and graph
        llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
        compiled_graph = agent_self_reflection.build_graph(llm)

        # Run the agent
        final_state = await compiled_graph.ainvoke({
            "user_input": user_input,
            "retry_count": 0,
            "validation_history": []
        })

        return {
            'status': 'success',
            'agent_type': 'Self-Reflective',
            'model': model_name,
            'parameters': final_state.get('parameters', {}),
            'simulation_results': final_state.get('simulation_result', {}),
            'formatted_parameters': final_state.get('formatted_parameters', ''),
            'validation_history': final_state.get('validation_history', []),
            'retry_count': final_state.get('retry_count', 0),
            'user_input': user_input,
            'debug_mode': debug_mode
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'agent_type': 'Self-Reflective'
        }

def format_results_for_chat(results: Dict[str, Any]) -> str:
    """Format simulation results for display in chat"""
    if results.get('status') != 'success':
        return f"❌ **Error:** {results.get('error', 'Unknown error')}"

    # Format the response
    response = f"✅ **Simulation Complete!**\n\n"

    # Add agent info
    response += f"**Agent:** {results['agent_type']}\n"
    if 'model' in results:
        response += f"**Model:** {results['model']}\n"
    elif 'models' in results:
        response += f"**Planning Model:** {results['models']['planning']}\n"
        response += f"**Summary Model:** {results['models']['summary']}\n"

    # Add validation info for self-reflection agent
    if results.get('retry_count', 0) > 0:
        response += f"**Validation Attempts:** {results['retry_count'] + 1}\n"

    response += "\n"

    # Format key metrics
    # sim_results = results.get('simulation_results', {})

    # if sim_results:
    #     response += "**📊 Key Performance Indicators:**\n"
    #     response += f"• **Mean Wait Time:** {sim_results.get('01_mean_waiting_time', 0):.2f} minutes\n"
    #     response += f"• **Operator Utilization:** {sim_results.get('02_operator_util', 0):.1f}%\n"
    #     response += f"• **Nurse Wait Time:** {sim_results.get('03_mean_nurse_waiting_time', 0):.2f} minutes\n"
    #     response += f"• **Nurse Utilization:** {sim_results.get('04_nurse_util', 0):.1f}%\n"
    #     response += f"• **Callback Rate:** {sim_results.get('05_callback_rate', 0):.1f}%\n\n"


    # Format parameters
    params = results.get('parameters', {})
    if params:
        response += "**⚙️ Parameters Used:**\n"
        for key, value in params.items():
            clean_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                response += f"• **{clean_key}:** {value:.3f}\n"
            else:
                response += f"• **{clean_key}:** {value}\n"

    return response


def format_results_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Display simulation results to the user in Streamlit chat
    """
    sim_results = results.get('simulation_results', {})
    df = pd.DataFrame(
        list(sim_results.items()), columns=["KPIs", "Values"]
    )
    
    return df


def display_debug_info(results: Dict[str, Any]) -> str:
    """Format debug information for chat display"""
    if not results.get('debug_mode'):
        return ""

    debug_info = "\n**🐛 Debug Information:**\n"

    if results['agent_type'] == 'Self-Reflective':
        validation_history = results.get('validation_history', [])
        if validation_history:
            debug_info += "\n**Validation History:**\n"
            for i, attempt in enumerate(validation_history, 1):
                status = "✅ Success" if attempt.get('validation_result', {}).get('is_valid', False) else "❌ Failed"
                debug_info += f"Attempt {i}: {status}\n"

                errors = attempt.get('validation_result', {}).get('errors', [])
                if errors:
                    debug_info += "Errors: " + ", ".join(errors) + "\n"

    return debug_info

# Header
st.markdown("# 🤖 Simulation Experimentation Agent Interface")

st.markdown(
    """**Ask the agent to perform simulation questions in natural language and it will provide you with results**.  
    Click the buttons for example questions.""")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Agent selection
agent_type = st.sidebar.selectbox(
    "Select Agent Type",
    ["Self-Reflective", "Dynamic Planner"],  # Prioritize working agent
    help="Choose between the two agent architectures"
)

# Model configuration
st.sidebar.subheader("🧠 LLM Configuration")

# Get models from Ollama with fallbacks
available_models = get_ollama_models()
if not available_models:
    available_models = ["gemma3:27b", "llama3:7b", "llama3:latest", "mistral:7b"]

# Find default indices
try:
    gemma_index = available_models.index("gemma3:27b")
except ValueError:
    gemma_index = 0

try:
    llama_index = available_models.index("llama3:latest")
except ValueError:
    llama_index = 0

if agent_type == "Dynamic Planner":
    planning_model = st.sidebar.selectbox(
        "Planning Model",
        available_models,
        index=gemma_index,
        help="Model used for reasoning and task planning"
    )

    summary_model = st.sidebar.selectbox(
        "Summary Model", 
        available_models,
        index=llama_index,
        help="Model used for summarizing parameters and results"
    )

    debug_mode = st.sidebar.checkbox(
        "Enable Debug Mode",
        help="Show detailed workflow and planning information"
    )

else:  # Self-Reflective Agent
    reflection_model = st.sidebar.selectbox(
        "Model",
        available_models,
        index=gemma_index,
        help="Model used for parameter generation and self-reflection"
    )

    debug_mode = st.sidebar.checkbox(
        "Enable Debug Mode",
        help="Show detailed reflection and retry information"
    )

# Server status
st.sidebar.subheader("🌐 Server Status")
mcp_status = check_server_status("http://localhost:8001/mcp", server_type="mcp") 
ollama_status = check_server_status("http://localhost:11434/api/tags", server_type="ollama")

st.sidebar.markdown(f"**MCP Server:** {'🟢' if mcp_status == 'Running' else '🔴'} {mcp_status}")
st.sidebar.markdown(f"**Ollama Server:** {'🟢' if ollama_status == 'Running' else '🔴'} {ollama_status}")

if mcp_status != "Running":
    st.sidebar.warning("⚠️ MCP Server not detected. Start with: `python mcp_server.py`")
if ollama_status != "Running":
    st.sidebar.warning("⚠️ Ollama Server not detected. Start with: `ollama serve`")

# Clear chat button in sidebar
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_messages = []
    st.session_state.simulation_results = None
    st.rerun()

# Main chat interface
st.subheader(f"💬 Chat with {agent_type} Agent")

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if not st.session_state.agent_busy:
    # Quick examples
    st.subheader("💡 Quick Examples")
    example_cols = st.columns(3)

    examples = [
        "Simulate 14 operators, 12 nurses and 5% extra demand",
        "Run scenario with high staffing and normal call volume", 
        "Test minimal configuration (5 operators, 3 nurses)"
    ]

    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Add example to chat and process
                st.session_state.chat_messages.append({"role": "user", "content": example})
                st.session_state.agent_busy = True
                st.rerun()

# Chat input
if prompt := st.chat_input("Enter your simulation request...", disabled=st.session_state.agent_busy):
    # Add user message to chat
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.agent_busy = True
    st.rerun()

# Process the latest message if agent is busy
if st.session_state.agent_busy and st.session_state.chat_messages:
    latest_message = st.session_state.chat_messages[-1]

    if latest_message["role"] == "user":
        user_input = latest_message["content"]

        # Check if servers are ready
        if mcp_status != "Running" or ollama_status != "Running":
            error_msg = "❌ **System Not Ready**\n\nPlease ensure both MCP and Ollama servers are running:\n"
            error_msg += "- MCP Server: `python mcp_server.py`\n"
            error_msg += "- Ollama Server: `ollama serve`"

            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
            st.session_state.agent_busy = False
            st.rerun()

        else:
            # Show processing message
            with st.chat_message("assistant"):
                with st.spinner(f"🤖 {agent_type} Agent is processing your request..."):

                    if agent_type == "Self-Reflective" and AGENTS_AVAILABLE:
                        # Run actual self-reflection agent
                        try:
                            start_time = time.time()

                            # Create progress indicator
                            progress_placeholder = st.empty()
                            progress_placeholder.info("🔄 Initializing self-reflection agent...")

                            # Run the agent
                            result = asyncio.run(run_self_reflection_agent(
                                user_input, reflection_model, debug_mode
                            ))

                            execution_time = time.time() - start_time
                            result['execution_time'] = execution_time

                            progress_placeholder.success(f"✅ Completed in {execution_time:.1f} seconds")

                        except Exception as e:
                            result = {
                                'status': 'error',
                                'error': str(e),
                                'agent_type': 'Self-Reflective'
                            }

                    else:
                        # Mock execution for Dynamic Planner or when agents not available
                        time.sleep(3)  # Simulate processing
                        result = {
                            'status': 'success',
                            'agent_type': agent_type,
                            'model': reflection_model if agent_type == "Self-Reflective" else None,
                            'models': {'planning': planning_model, 'summary': summary_model} if agent_type == "Dynamic Planner" else None,
                            'parameters': {
                                'n_operators': 14, 'n_nurses': 12, 'mean_iat': 0.57,
                                'call_low': 5.0, 'call_mode': 7.0, 'call_high': 10.0,
                                'callback_prob': 0.4, 'run_length': 1000, 'random_seed': 42
                            },
                            'simulation_result': {
                                '01_mean_waiting_time': 2.34, '02_operator_util': 76.8,
                                '03_mean_nurse_waiting_time': 1.45, '04_nurse_util': 68.2,
                                '05_callback_rate': 41.2
                            },
                            'retry_count': 0,
                            'debug_mode': debug_mode
                        }

            # Format and add response
            response_text = format_results_for_chat(result)
            st.markdown(response_text)

            # Display results table if simulation results exist
            sim_results = result.get('simulation_results', {})   
            if 'simulation_results' in result and result['simulation_results']:
                st.write("\n")  # Add spacing
                df = pd.DataFrame(
                    list(sim_results.items()), 
                    columns=["KPIs", "Values"]
                )

                response_text += "\n**📊 Key Performance Indicators:**\n"
                st.markdown("**📊 Key Performance Indicators:**\n")
                #st.dataframe(df.round(2), width='stretch')
                st.markdown(df.round(2).to_markdown(index=False))
            
            # Add debug info if enabled
            if debug_mode:
                debug_info = display_debug_info(result)
                st.markdown(debug_info)

            # Store message content for history (text only, no table)
            response_text_stored = format_results_for_chat(result)
            response_text_stored += "\n\n📊 **Key Performance Indicators:**\n"
            response_text_stored += df.round(2).to_markdown(index=False)
            if debug_mode:
                response_text_stored += display_debug_info(result)
            response_text_stored += f"\n\n*Completed at {datetime.now().strftime('%H:%M:%S')}*"
            
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": response_text_stored,
                "has_table": True,
                "table_data": sim_results
            })
            st.session_state.simulation_results = result
            st.session_state.agent_busy = False
            st.rerun()

# Export options
if st.session_state.simulation_results:
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    results = st.session_state.simulation_results

    with col1:
        if results.get('parameters'):
            param_df = pd.DataFrame([
                {"Parameter": k.replace('_', ' ').title(), "Value": v}
                for k, v in results['parameters'].items()
            ])
            csv_params = param_df.to_csv(index=False)
            st.download_button(
                "📊 Parameters CSV",
                csv_params,
                file_name=f"parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col2:
        if results.get('simulation_results'):
            results_df = pd.DataFrame([
                {"KPI": k.replace('_', ' ').title(), "Value": v}
                for k, v in results['simulation_results'].items()
            ])
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                "📈 Results CSV",
                csv_results,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col3:
        full_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📋 Full Report JSON",
            full_json,
            file_name=f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

