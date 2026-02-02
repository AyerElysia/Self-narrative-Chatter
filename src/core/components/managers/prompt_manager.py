"""
Prompt manager for components module.

This module re-exports the PromptManager from the prompt module for use in the components system.
"""

from src.core.prompt import PromptManager, get_prompt_manager

__all__ = ["PromptManager", "get_prompt_manager"]
