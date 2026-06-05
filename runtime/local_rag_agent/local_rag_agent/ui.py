from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DemoSource:
    label: str
    source: str = ""


@dataclass(frozen=True)
class WelcomeItem:
    title: str
    text: str


@dataclass(frozen=True)
class UiConfig:
    title: str = "Local Agent"
    home_title: str = "Local Agent Workspace"
    home_heading: str = "Agent workspace"
    home_lead: str = "Open the local agent, inspect sources, and run project-specific regression checks."
    placeholder: str = "Ask this agent"
    status_text: str = "Searching the project knowledge base and preparing an answer..."
    welcome_intro: str = "This local agent answers from the configured project knowledge base and shows sources for review."
    welcome_items: list[WelcomeItem] = field(default_factory=list)
    demo_sources: list[DemoSource] = field(default_factory=list)


def load_ui_config(path: Path | None) -> UiConfig:
    if path is None or not path.exists():
        return UiConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    return UiConfig(
        title=str(data.get("title", "Local Agent")),
        home_title=str(data.get("home_title", "Local Agent Workspace")),
        home_heading=str(data.get("home_heading", "Agent workspace")),
        home_lead=str(data.get("home_lead", "Open the local agent, inspect sources, and run project-specific regression checks.")),
        placeholder=str(data.get("placeholder", "Ask this agent")),
        status_text=str(data.get("status_text", "Searching the project knowledge base and preparing an answer...")),
        welcome_intro=str(
            data.get(
                "welcome_intro",
                "This local agent answers from the configured project knowledge base and shows sources for review.",
            )
        ),
        welcome_items=[
            WelcomeItem(title=str(item.get("title", "")), text=str(item.get("text", "")))
            for item in data.get("welcome_items", [])
            if isinstance(item, dict)
        ],
        demo_sources=[
            DemoSource(label=str(item.get("label", "")), source=str(item.get("source", "")))
            for item in data.get("demo_sources", [])
            if isinstance(item, dict)
        ],
    )
