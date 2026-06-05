from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .adapters.generators import ExtractiveGenerator, OpenAICompatibleGenerator
from .adapters.retrievers import LexicalRetriever
from .config import Settings
from .policy import PolicyGuard
from .ports import GeneratorPort, GeneratorProvider, RetrieverPort, RetrieverProvider
from .tools import ToolProvider
from .workflows.runner import WorkflowStep
from .workflows.steps import StepRegistry

RetrieverFactory = Callable[[Settings], RetrieverPort]
GeneratorFactory = Callable[[Settings], GeneratorPort]
PolicyFactory = Callable[[Settings], object]
ToolFactory = Callable[[Settings], object]


@dataclass(frozen=True)
class StepDefinition:
    id: str
    fn: WorkflowStep
    description: str = ""
    terminal: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    timeout_seconds: float | None = None
    checkpoint_after: bool = False


class ComponentRegistry:
    def __init__(self):
        self._steps: dict[str, WorkflowStep] = {}
        self._step_definitions: dict[str, StepDefinition] = {}
        self._retrievers: dict[str, RetrieverFactory] = {}
        self._generators: dict[str, GeneratorFactory] = {}
        self._policy_providers: dict[str, PolicyFactory] = {}
        self._tool_providers: dict[str, ToolFactory] = {}
        self._trace_sinks: dict[str, object] = {}

    @classmethod
    def builtins(cls) -> "ComponentRegistry":
        registry = cls()
        for step_id, step in StepRegistry.builtins().steps.items():
            registry.register_step_definition(
                StepDefinition(
                    id=step_id,
                    fn=step,
                    terminal=step_id
                    in {
                        "build_policy_response",
                        "build_response",
                        "build_retrieval_debug_response",
                        "build_refusal_response",
                        "response.tool_result",
                    },
                )
            )
        registry.register_retriever("lexical", lambda settings: LexicalRetriever())
        registry.register_generator("extractive", lambda settings: ExtractiveGenerator())
        registry.register_generator("openai_compatible", lambda settings: OpenAICompatibleGenerator())
        registry.register_policy_provider("keyword", lambda settings: PolicyGuard.from_config(settings.policy_config_path))
        registry.register_tool_provider("configured", lambda settings: ToolProvider.from_config(settings.tool_config_path))
        return registry

    @classmethod
    def from_settings(cls, settings: Settings) -> "ComponentRegistry":
        registry = cls.builtins()
        registry.load_plugins(settings)
        return registry

    def load_plugins(self, settings: Settings) -> None:
        for module_name in settings.plugin_modules:
            self.load_plugin_module(module_name, settings)

    def load_plugin_module(self, module_name: str, settings: Settings) -> None:
        project_root = str(settings.project_root)
        inserted = False
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            inserted = True
        try:
            importlib.invalidate_caches()
            self._clear_plugin_module_cache(module_name)
            module = importlib.import_module(module_name)
        finally:
            if inserted:
                try:
                    sys.path.remove(project_root)
                except ValueError:
                    pass
        register = getattr(module, "register", None)
        if not callable(register):
            raise ValueError(f"Plugin module must expose register(registry): {module_name}")
        register(self)

    def register_step(self, name: str | StepDefinition, step: WorkflowStep | None = None) -> None:
        if isinstance(name, StepDefinition):
            self.register_step_definition(name)
            return
        if step is None:
            raise ValueError(f"Step registration requires a callable: {name}")
        self.register_step_definition(StepDefinition(id=name, fn=step))

    def register_step_definition(self, definition: StepDefinition) -> None:
        if definition.id in self._steps:
            raise ValueError(f"Duplicate component registration: step {definition.id}")
        self._steps[definition.id] = definition.fn
        self._step_definitions[definition.id] = definition

    def has_step(self, name: str) -> bool:
        return name in self._steps

    def get_step(self, name: str) -> WorkflowStep:
        return self._get(self._steps, "step", name)

    def get_step_definition(self, name: str) -> StepDefinition:
        return self._get(self._step_definitions, "step", name)

    def step_definitions(self) -> dict[str, StepDefinition]:
        return dict(self._step_definitions)

    def terminal_steps(self) -> set[str]:
        return {
            step_id
            for step_id, definition in self._step_definitions.items()
            if definition.terminal
        }

    def step_registry(self) -> StepRegistry:
        return StepRegistry(dict(self._steps))

    def register_retriever(self, name: str, factory: RetrieverFactory) -> None:
        self._register(self._retrievers, "retriever", name, factory)

    def get_retriever(self, name: str) -> RetrieverFactory:
        return self._get(self._retrievers, "retriever", name)

    def build_retriever_provider(self, settings: Settings) -> RetrieverProvider:
        return RetrieverProvider(self.get_retriever(settings.retrieval_provider)(settings))

    def register_generator(self, name: str, factory: GeneratorFactory) -> None:
        self._register(self._generators, "generator", name, factory)

    def get_generator(self, name: str) -> GeneratorFactory:
        return self._get(self._generators, "generator", name)

    def build_generator_provider(self, settings: Settings) -> GeneratorProvider:
        return GeneratorProvider(self.get_generator(settings.generation_provider)(settings))

    def register_policy_provider(self, name: str, factory: PolicyFactory) -> None:
        self._register(self._policy_providers, "policy_provider", name, factory)

    def get_policy_provider(self, name: str) -> PolicyFactory:
        return self._get(self._policy_providers, "policy_provider", name)

    def build_policy_guard(self, settings: Settings) -> object:
        return self.get_policy_provider("keyword")(settings)

    def register_tool_provider(self, name: str, factory: ToolFactory) -> None:
        self._register(self._tool_providers, "tool_provider", name, factory)

    def get_tool_provider(self, name: str) -> ToolFactory:
        return self._get(self._tool_providers, "tool_provider", name)

    def build_tool_provider(self, settings: Settings) -> object:
        return self.get_tool_provider("configured")(settings)

    def register_trace_sink(self, name: str, sink: object) -> None:
        self._register(self._trace_sinks, "trace_sink", name, sink)

    def get_trace_sink(self, name: str) -> object:
        return self._get(self._trace_sinks, "trace_sink", name)

    def emit_trace(self, event: dict[str, object]) -> None:
        for sink in self._trace_sinks.values():
            emit = getattr(sink, "emit", None)
            if callable(emit):
                emit(event)
            elif callable(sink):
                sink(event)

    @staticmethod
    def _clear_plugin_module_cache(module_name: str) -> None:
        parts = module_name.split(".")
        for index in range(len(parts), 0, -1):
            sys.modules.pop(".".join(parts[:index]), None)

    @staticmethod
    def _register(bucket: dict[str, object], kind: str, name: str, value: object) -> None:
        if name in bucket:
            raise ValueError(f"Duplicate component registration: {kind} {name}")
        bucket[name] = value

    @staticmethod
    def _get(bucket: dict[str, object], kind: str, name: str) -> object:
        if name not in bucket:
            raise KeyError(f"Missing component registration: {kind} {name}")
        return bucket[name]
