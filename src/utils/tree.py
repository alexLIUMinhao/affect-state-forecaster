from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _induced_parent_map(
    thread_id: str,
    conversation_tree: dict[str, str | None] | None,
    observed_replies: list[dict[str, Any]],
) -> dict[str, str | None]:
    allowed_nodes = {thread_id}
    allowed_nodes.update(str(reply.get("id", "")) for reply in observed_replies if reply.get("id") is not None)

    parent_map: dict[str, str | None] = {thread_id: None}
    tree = conversation_tree or {}
    for node_id, parent_id in tree.items():
        node_key = str(node_id)
        parent_key = str(parent_id) if parent_id is not None else None
        if node_key in allowed_nodes:
            parent_map[node_key] = parent_key if parent_key in allowed_nodes else None

    for reply in observed_replies:
        node_id = str(reply.get("id", ""))
        parent_id = reply.get("parent_id")
        parent_key = str(parent_id) if parent_id is not None and str(parent_id) in allowed_nodes else thread_id
        parent_map.setdefault(node_id, parent_key)

    return parent_map


def compute_tree_statistics(
    thread_id: str,
    conversation_tree: dict[str, str | None] | None,
    observed_replies: list[dict[str, Any]],
) -> np.ndarray:
    """Compute lightweight structure features for an observed conversation subtree."""

    parent_map = _induced_parent_map(thread_id, conversation_tree, observed_replies)
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, parent_id in parent_map.items():
        if parent_id is not None:
            children[parent_id].append(node_id)

    depths: dict[str, int] = {thread_id: 0}
    stack = [thread_id]
    while stack:
        node_id = stack.pop()
        for child_id in children.get(node_id, []):
            depths[child_id] = depths[node_id] + 1
            stack.append(child_id)

    reply_ids = [str(reply.get("id", "")) for reply in observed_replies if reply.get("id") is not None]
    reply_depths = [depths.get(node_id, 1) for node_id in reply_ids]
    node_count = len(parent_map)
    reply_count = len(reply_ids)
    edge_count = max(node_count - 1, 0)
    root_children = len(children.get(thread_id, []))
    internal_nodes = sum(1 for node_id in parent_map if children.get(node_id))
    leaves = sum(1 for node_id in parent_map if not children.get(node_id))
    avg_depth = float(np.mean(reply_depths)) if reply_depths else 0.0
    max_depth = float(max(reply_depths)) if reply_depths else 0.0
    branching_factor = edge_count / max(internal_nodes, 1)
    leaf_ratio = leaves / max(node_count, 1)

    return np.asarray(
        [
            float(reply_count),
            float(node_count),
            float(edge_count),
            float(root_children),
            avg_depth,
            max_depth,
            float(branching_factor),
            float(leaf_ratio),
        ],
        dtype=np.float32,
    )
