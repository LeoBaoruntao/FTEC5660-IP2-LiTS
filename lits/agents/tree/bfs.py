from collections import defaultdict
from dataclasses import dataclass, field, asdict
import json
import logging

from .node import SearchNode
from .base import BaseSearchConfig
from .search_base import BaseTreeSearch, SearchResult
from .continuation import _continuation
from .common import _world_modeling, _is_terminal_with_depth_limit, _sample_actions_with_existing, create_child_node
from ..registry import register_search
from ... import observability as tracing
from ...log import log_phase, log_event
from ...visualize import visualize_tree

logger = logging.getLogger(__name__)

@dataclass
class BFSConfig(BaseSearchConfig):
    """BFS-specific search configuration.
    
    Config Args (via --search-arg):
        beam_size: Number of top candidates to keep at each depth level (default: 5)
        max_leaves_to_terminate: Stop when counted terminal endpoints reach this threshold
            (default: 5). Checks follow expansion batches, so the count can overshoot.
    """
    beam_size: int = 5
    max_leaves_to_terminate: int = 5
    
    def to_dict(self):
        return asdict(self)

@dataclass
class BFSResult(SearchResult):
    """BFS-specific search result.

    Extends ``SearchResult`` with depth-bucketed node data.
    ``to_paths()`` delegates to ``buckets_to_paths()`` for
    breadth-to-depth conversion.
    Depth/repetition cutoffs remain in the tree and buckets, but are excluded
    from ``terminal_nodes_collected`` unless ``is_terminal=True``.
    """
    buckets_with_terminal: dict = field(default_factory=dict)

    def to_paths(self) -> list[list[SearchNode]]:
        """BFS paths: reconstruct from depth buckets."""
        from ...visualize import buckets_to_paths
        return buckets_to_paths(self.buckets_with_terminal)


##### EXPAND (Begin) #####
def _expand(
    example,
    query_idx,
    node,
    policy,
    n_actions,
    world_model=None,
    reward_model=None,
    assign_rewards=True,
    from_phase=""
):
    """
    Expand the node with new actions. 
    """
    log_phase(logger, "Expand", "Begin")
    steps = policy.get_actions(node.state, query=example, n_actions=n_actions, query_idx=query_idx, from_phase=from_phase)

    is_terminal_for_repeats = []
    for step in steps:
        action = step.get_action()  # Extract action from Step object
        # Mark as terminal if action is repeat sentinel OR step has terminate flag
        is_terminal = (action == "ALWAY REPEAT. TERMINATE") or getattr(step, 'terminate', False)
        is_terminal_for_repeats.append(is_terminal)

    # Determine the starting index for new children (to handle existing children)
    existing_children_count = len(node.children)

    for idx, (step, is_terminal_for_repeat) in enumerate(zip(steps, is_terminal_for_repeats)):
        action = step.get_action()  # Extract action from Step object
        child_idx = existing_children_count + idx
        
        # Use unified helper to create child with proper trajectory_key
        child = create_child_node(
            SearchNode,
            parent=node,
            action=action,
            step=step,
            child_index=child_idx
        )
        child.is_terminal_for_repeat = is_terminal_for_repeat
        
        # Assign fast_reward using common helper
        if assign_rewards:
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)
        
        node.children.append(child)    
    log_phase(logger, "Expand", "End")
##### EXPAND (END) #####


##### EXPAND With Existing Children (BEGIN) #####
def _expand_with_existing(
    example,
    query_idx,
    node,
    policy,
    n_actions,
    reward_model=None,
    world_model=None,
    assign_rewards=True,
    from_phase="",
):
    """ Expand the node with existing children. 
    This is designed for BFS with continuous phase but compatible for the original BFS. """
    log_phase(logger, "Expand", f"Begin (example={query_idx})")

    new_actions = _sample_actions_with_existing(
        example,
        query_idx,
        node,
        policy,
        n_actions,
        from_phase=from_phase,
    )

    # Determine the starting index for new children (to handle existing children).
    # This ensures each child gets a unique trajectory_key index.
    existing_children_count = len(node.children)

    # Step 3: Assign rewards + terminal flags for new actions
    for idx, step in enumerate(new_actions):
        action = step.get_action()  # Extract action from Step object
        child_idx = existing_children_count + idx
        
        # Use unified helper to create child with proper trajectory_key
        child = create_child_node(
            SearchNode,
            parent=node,
            action=action,
            step=step,
            child_index=child_idx
        )

        # Assign terminal-for-repeat: check both repeat sentinel and step.terminate flag
        child.is_terminal_for_repeat = (action == "ALWAY REPEAT. TERMINATE") or getattr(step, 'terminate', False)

        # Assign fast_reward using common helper
        if assign_rewards and (child.fast_reward == -1):
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)

        node.children.append(child)

    # Step 4: Ensure existing children have the required attributes
    for child in node.children:

        if assign_rewards and (child.fast_reward == -1):
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)

    log_phase(logger, "Expand", f"End (example={query_idx})")
##### EXPAND With Existing Children (END) #####


##### BFS (BEGIN) #####
@register_search("bfs", config_class=BFSConfig)
class BFSSearch(BaseTreeSearch):
    """BFS tree search algorithm as a ``BaseTreeSearch`` subclass.

    Peripherals (node ID reset, root creation, checkpoint dir, runtime
    tracking, terminal collection, error handling, inference logger) are
    handled by ``BaseTreeSearch``.  This class implements the depth-bucketed
    frontier loop with beam pruning.

    Extension via subclassing
    -------------------------
    Override ``_do_expand(...)`` to customize expansion behavior.
    Default delegates to module-level ``_expand_with_existing()``.
    ``_continuation`` receives ``self._do_expand`` as ``expand_func``.

    Override ``_do_select_beam(...)`` to customize beam selection.
    The default preserves descending ``fast_reward`` pruning.

    See ``docs/agents/tree/mcts/MCTS_SEARCH_LOOP.md`` for the extension
    pattern (same approach as MCTS).
    """

    node_class = SearchNode

    def __init__(self, *args, observations: dict | None = None, **kwargs):
        """Initialize ordinary BFS with optional caller-owned display rules.

        All other arguments pass unchanged to BaseTreeSearch. ``observations``
        is passed as ``settings`` to ``tracing.observe``; it is not a second
        configuration. Omit it to skip explicit snapshots. For example::

            observations = {
                "node": {"name": "search.node", "fields": ("id", "state", "score")},
                "frontier": {"name": "search.frontier", "fields": ("when", "node_ids", "scores")},
                "score": {"name": "search.score", "fields": ("node_id", "score", "details")},
            }
            search = BFSSearch(config, world_model, policy, reward_model,
                               observations=observations)
            result = search.run(query, query_idx=0)

        Node fields also include serialized action/step, parent_id, depth, and
        terminal/repetition flags. Frontier ``when`` is before_selection or
        after_selection. Score ``details`` comes from the existing fast reward.
        The caller configures/closes tracing; rules never affect search decisions.
        Root/iteration spans and model tracing do not require snapshot rules.
        """
        super().__init__(*args, **kwargs)
        self.observations = observations if observations is not None else {}

    def _observe_node(self, node: SearchNode) -> None:
        """Snapshot existing node data and local reward without evaluating again."""
        if "node" in self.observations:
            tracing.observe("node", {
                **node.to_dict(),
                "parent_id": node.parent.id if node.parent is not None else None,
                "depth": node.depth,
                "score": node.fast_reward if node.parent is not None else None,
                "is_terminal_for_repeat": node.is_terminal_for_repeat,
            }, settings=self.observations)
        if "score" in self.observations and node.parent is not None:
            tracing.observe("score", {
                "node_id": node.id,
                "score": node.fast_reward,
                "details": getattr(node, "fast_reward_details", {}),
            }, settings=self.observations)

    def _observe_frontier(self, frontier: list[SearchNode], when: str) -> None:
        """Snapshot candidate order and scores on either side of beam selection."""
        if "frontier" in self.observations:
            tracing.observe("frontier", {
                "when": when,
                "node_ids": [node.id for node in frontier],
                "scores": [node.fast_reward for node in frontier],
            }, settings=self.observations)

    def _do_expand(self, query_or_goals, query_idx, node, policy, n_actions, **kwargs):
        """Expand phase — override in subclasses for custom expansion.

        Default delegates to the module-level ``_expand_with_existing()``.
        """
        _expand_with_existing(query_or_goals, query_idx, node, policy, n_actions, **kwargs)

    def _do_select_beam(self, query, query_idx, depth, frontier):
        """Return the nodes retained at the current depth.

        The default computes missing fast rewards, sorts candidates by
        descending reward, and retains at most ``config.beam_size`` nodes.
        Scoring receives the full step so policy scores and other metadata survive.
        Subclasses may override this method to implement another selector.
        """
        if len(frontier) <= self.config.beam_size:
            return frontier

        for node in frontier:
            if node.fast_reward == -1 or node.fast_reward is None:
                logger.debug(f"Fast reward not computed for node {node.action} for sort")
                fast_reward, _ = self.reward_model.fast_reward(
                    node.parent.state,
                    node.step,
                    query,
                    query_idx,
                    from_phase="sort",
                )
                node.fast_reward = fast_reward

        frontier.sort(key=lambda n: n.fast_reward, reverse=True)
        return frontier[: self.config.beam_size]

    def search(self, query, query_idx) -> BFSResult:
        """Run BFS iterations. ``self.root`` is ready.

        One search.run span contains the existing depth loop; each processed
        depth gets a search.iteration span containing its calls and snapshots.
        Selection, expansion, terminal collection, and checkpoints are unchanged.
        """
        with tracing.span("search.run", algorithm="bfs", query=query, query_idx=query_idx):
            logger.debug(f"Question: {query}")
            log_phase(logger, "BFS", f"Begin (example={query_idx})")

            self._observe_node(self.root)
            config = self.config
            stop_continuation = False
            terminal_nodes = []

            # Per-depth buckets: absolute_depth -> list[SearchNode]
            frontier_buckets = defaultdict(list)
            frontier_buckets[0].append(self.root)

            buckets_with_terminal = defaultdict(list)
            buckets_with_terminal[0].append(self.root)

            for depth in range(config.max_steps):
                # Use shared check but catch locally so the normal
                # terminal-collection logic below still runs with buckets intact.
                try:
                    self.check_runtime_limit()
                except ValueError:
                    log_event(logger, "BFS", f"Runtime limit exceeded: {config.runtime_limit_before_iter}", level="debug")
                    break
                with tracing.span("search.iteration", depth=depth):
                    log_phase(logger, "BFS", f"Depth {depth} Begin")

                    # Set iteration (depth) field for all LLM calls at this depth
                    self.set_log_field("iteration", depth)

                    # Define callback to update trajectory_key at each hop
                    def update_traj_key(node):
                        if node.trajectory_key:
                            self.set_log_field("trajectory_key", node.trajectory_key.path_str)

                    # 1) Take all candidates scheduled at this depth, then beam-prune
                    frontier = frontier_buckets.get(depth, [])
                    if not frontier:
                        log_event(logger, "BFS", "No nodes at this depth, breaking", level="debug")
                        break

                    # Beam pruning on current layer
                    self._observe_frontier(frontier, "before_selection")
                    frontier = self._do_select_beam(query, query_idx, depth, frontier)
                    self._observe_frontier(frontier, "after_selection")

                    # 2) Loop each node in the frontier at this depth (Begin)
                    for node in frontier:
                        # Update trajectory_key for current node
                        update_traj_key(node)

                        if _is_terminal_with_depth_limit(node, config.max_steps, config.force_terminating_on_depth_limit):
                            if node not in terminal_nodes:
                                terminal_nodes.append(node)
                            continue
                        if len(node.children) > 0:  # branching is done in the previous continuation or expand
                            continue
                        if len(terminal_nodes) >= config.max_leaves_to_terminate:
                            log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking", level="debug")
                            break

                        # Ensure node.state is materialized
                        if node.state is None:
                            _world_modeling(query, query_idx, node, transition_model=self.world_model, reward_model=self.reward_model, from_phase="expand")
                            self._observe_node(node)

                        # Continuation + PostProcessing (Begin)
                        if config.add_continuation and not stop_continuation:
                            if config.only_continuation_at_head:
                                stop_continuation = True
                            cont_trace = _continuation(
                                query,
                                query_idx,
                                node,
                                self.world_model,
                                self.policy,
                                self.reward_model,
                                expand_func=self._do_expand,
                                bn_evaluator=self.bn_evaluator,
                                world_modeling_func=_world_modeling,
                                threshold_alpha=config.reward_alpha,
                                threshold_conf=config.reward_beta,
                                threshold_gamma=config.reward_gamma,
                                threshold_gamma1=config.reward_gamma1,
                                n_actions_for_bne=config.n_actions_for_bne,
                                on_step=update_traj_key
                            )

                            # Place each continuation hop at correct future depth
                            # cont_trace[i] belongs to depth = depth + i
                            for i, cnode in enumerate(cont_trace[1:]):
                                assert cnode.state is not None, f"`_continuation` returns a node without materialized state"
                                self._observe_node(cnode)

                                if _is_terminal_with_depth_limit(cnode, config.max_steps, config.force_terminating_on_depth_limit):
                                    if cnode not in terminal_nodes:
                                        terminal_nodes.append(cnode)
                                    assert len(cont_trace[1:]) == i + 1, f"Continuation trace includes node(s) at the depth beyond the depth limit"
                                else:
                                    frontier_buckets[cnode.depth].append(cnode)
                                buckets_with_terminal[cnode.depth].append(cnode)
                            node = cont_trace[-1]

                            # Update trajectory_key after continuation
                            update_traj_key(node)

                            if _is_terminal_with_depth_limit(node, config.max_steps, config.force_terminating_on_depth_limit):
                                if node not in terminal_nodes:
                                    terminal_nodes.append(node)
                                continue
                            if len(terminal_nodes) >= config.max_leaves_to_terminate:
                                log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking (after continuation)", level="debug")
                                break
                        # Continuation + PostProcessing (End)

                        assert node.state is not None
                        self._do_expand(
                            query,
                            query_idx,
                            node,
                            self.policy,
                            config.n_actions,
                            reward_model=self.reward_model,
                            from_phase="expand"
                        )
                        for child in node.children:
                            _world_modeling(query, query_idx, child, transition_model=self.world_model, reward_model=self.reward_model, from_phase="expand")
                            self._observe_node(child)
                            if _is_terminal_with_depth_limit(child, config.max_steps, config.force_terminating_on_depth_limit):
                                if child not in terminal_nodes:
                                    terminal_nodes.append(child)
                            else:
                                frontier_buckets[child.depth].append(child)
                            buckets_with_terminal[child.depth].append(child)

                        if len(terminal_nodes) >= config.max_leaves_to_terminate:
                            log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking (after expand)", level="debug")
                            break
                    # 2) Loop each node in the frontier at this depth (End)
                    if len(terminal_nodes) >= config.max_leaves_to_terminate:
                        log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking (end of depth)", level="debug")
                        break
                    log_phase(logger, "BFS", f"Depth {depth} End")

                    # Log tree snapshot after this layer
                    tree_str = visualize_tree(self.root)
                    logger.info(f"[BFS] Depth {depth}/{config.max_steps} (example={query_idx}) | terminals={len(terminal_nodes)}\n{tree_str}")

                    # Save level-wise checkpoint
                    if self._checkpoint_path:
                        from ...structures.trace import _serialize_obj
                        depth_nodes = [_serialize_obj(n) for n in buckets_with_terminal.get(depth, [])]
                        self.save_checkpoint(query_idx, depth, depth_nodes)

            # Collect all terminal nodes from various sources
            # The loop also counts cutoff endpoints; evaluation receives only task terminals.
            terminal_nodes_collected = [node for node in terminal_nodes if node.is_terminal]

            # Check frontier for additional terminal nodes
            for node in frontier:
                if node.is_terminal and node not in terminal_nodes_collected:
                    terminal_nodes_collected.append(node)

            # Check deepest bucket for terminal nodes
            if buckets_with_terminal:
                max_d = max(buckets_with_terminal.keys())
                log_event(logger, "BFS", f"Frontier candidates at depth {max_d}: {len(buckets_with_terminal[max_d])}", level="debug")
                for n in buckets_with_terminal[max_d]:
                    if n.is_terminal and n not in terminal_nodes_collected:
                        terminal_nodes_collected.append(n)

            log_event(logger, "BFS", f"Total terminal nodes collected: {len(terminal_nodes_collected)}", level="debug")
            log_phase(logger, "BFS", f"End (example={query_idx})")

            return BFSResult(
                root=self.root,
                terminal_nodes_collected=terminal_nodes_collected,
                buckets_with_terminal=buckets_with_terminal
            )

##### BFS (END) #####
