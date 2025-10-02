"""Singleton for CoordinatorAgent and all sub-agents."""
from src.agents.coordinator import CoordinatorAgent

# Create a single CoordinatorAgent instance for the whole app
coordinator_agent = CoordinatorAgent()