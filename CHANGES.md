# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Dates formatted as YYYY-MM-DD as per [ISO standard](https://www.iso.org/iso-8601-date-and-time-format.html).

Consistent identifier (represents all versions, resolves to latest): 

## [v0.1.0](https://github.com/pythonhealthdatascience/des_agent/releases/tag/v0.1.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16616715.svg)](https://doi.org/10.5281/zenodo.16616715)

Initial release. Feasibility of the approach. Simple commend line interface enhanced using `rich`. Tested with gemma3:27b and mistral:7b via Ollama. Running on RTX 4080.

### Added

* `model.py` - simple urgent care call centre model. Serves as the simulation layer.
* `mcp_server.py` - Model Context Protocol layer implemented using `FastMCP`. 
* `agent_self_reflection.py`: `LangGraph` state chart agent.  Self reflection fixes paramter generation mistakes
* `agent_planning_workflow.py`: LangChain implementation. Agent discovers servers abilities and reasons about the order in which to take action. 
* Repo fundamentals: citation info, license, change log, environment file, readme.

