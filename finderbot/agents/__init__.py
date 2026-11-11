"""
agents package initialization
"""
from .inspector_agent import ItemInspectorAgent
from .matcher_agent import ItemMatcherAgent

# Alias so legacy imports don't break
InspectorAgent = ItemInspectorAgent
MatcherAgent = ItemMatcherAgent

__all__ = [
    "ItemInspectorAgent",
    "InspectorAgent",
    "ItemMatcherAgent",
    "MatcherAgent",
]