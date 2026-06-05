from __future__ import annotations

from .conditions import condition_matches
from .runner import WorkflowContext, WorkflowStep, _write_checkpoint


class GraphWorkflow:
    def __init__(
        self,
        workflow_id: str,
        nodes: list[dict[str, object]],
        edges: list[dict[str, object]],
        start: str,
        step_map: dict[str, WorkflowStep],
        requires_sources: bool | None = None,
        max_steps: int = 50,
    ):
        self.workflow_id = workflow_id
        self.nodes = {str(node["id"]): node for node in nodes}
        self.edges = edges
        self.start = start
        self.step_map = step_map
        self.requires_sources = requires_sources
        self.max_steps = max_steps

    def run(self, context: WorkflowContext):
        context.workflow_requires_sources = self.requires_sources
        context.trace.add_step(
            "start_graph_workflow",
            {
                "workflow": self.workflow_id,
                "start": self.start,
                "node_count": len(self.nodes),
                "requires_sources": self.requires_sources,
            },
        )
        current = self.start
        visited = 0
        while current:
            visited += 1
            if visited > self.max_steps:
                raise RuntimeError(f"Graph workflow exceeded max steps: {self.workflow_id}")
            node = self.nodes[current]
            step_id = str(node["step"])
            step = self.step_map[step_id]
            step(context)
            if bool(node.get("checkpoint_after")):
                _write_checkpoint(context, current)
            if context.response is not None or bool(node.get("terminal")):
                break
            current = self._next_node(current, context)
        if context.response is None:
            raise RuntimeError(f"Workflow did not produce a response: {self.workflow_id}")
        return context.response

    def _next_node(self, current: str, context: WorkflowContext) -> str:
        candidates = [edge for edge in self.edges if edge.get("from") == current]
        for edge in candidates:
            condition = str(edge.get("condition", "default"))
            if condition != "default" and condition_matches(condition, context):
                return str(edge.get("to", ""))
        for edge in candidates:
            if condition_matches(str(edge.get("condition", "default")), context):
                return str(edge.get("to", ""))
        return ""
