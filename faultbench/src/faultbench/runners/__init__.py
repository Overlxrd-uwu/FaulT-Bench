"""Agent runners — one module per evaluated agent (SADE, CC-Baseline,
ReAct), each pairing its free-text prompt with the Kathara tool layer
imported from the SADE-NetworkAgent sibling.
"""
from .base import AgentResult, RunnerError
