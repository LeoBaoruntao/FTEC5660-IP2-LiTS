"""
Test suite for extract_answers_from_terminal_nodes function.

Tests common.py::extract_answers_from_terminal_nodes and real terminal-node
deserialization. Current CLI and model-based voting checks live in
unit_test/eval/test_search_evaluation.py.

run:

```
python unit_test/agents/test_extract_answers.py
```
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import RAP formulation to register SubQAStep type
import lits_benchmark.formulations.rap  # noqa: F401

from lits.agents.tree.node import SearchNode, MCTSNode
from lits.agents.tree.common import extract_answers_from_terminal_nodes
from lits.components.utils import retrieve_answer_from_last_step


# =============================================================================
# Mock data for basic tests (kept from original)
# =============================================================================

def create_mock_tree_with_terminals():
    """Create a mock tree with terminal nodes for testing."""
    # Root node
    root = SearchNode(state=[], action="Root", parent=None)
    root.id = 0
    root.fast_reward = 0.0
    root.is_terminal = False
    
    # Terminal nodes with different rewards
    terminal1 = SearchNode(
        state=[{"action": "Answer: 42", "__type__": "Step"}],
        action="Answer: 42",
        parent=root
    )
    terminal1.id = 1
    terminal1.fast_reward = 0.9
    terminal1.is_terminal = True
    
    terminal2 = SearchNode(
        state=[{"action": "Answer: 43", "__type__": "Step"}],
        action="Answer: 43",
        parent=root
    )
    terminal2.id = 2
    terminal2.fast_reward = 0.7
    terminal2.is_terminal = True
    
    terminal3 = SearchNode(
        state=[{"action": "Answer: 42", "__type__": "Step"}],
        action="Answer: 42 (duplicate)",
        parent=root
    )
    terminal3.id = 3
    terminal3.fast_reward = 0.8
    terminal3.is_terminal = True
    
    root.children = [terminal1, terminal2, terminal3]
    
    return root, [terminal1, terminal2, terminal3]


def mock_retrieve_answer(state, question):
    """Mock answer retrieval function."""
    if state and len(state) > 0:
        action = state[-1].get("action", "")
        if "42" in action:
            return "42"
        elif "43" in action:
            return "43"
    return "unknown"


# =============================================================================
# Real terminal nodes data from math500_mcts run
# =============================================================================

REAL_TERMINAL_NODES_JSON = '''
{
  "terminal_nodes": [
    {
      "id": 4,
      "state": {
        "__type__": "TrajectoryState",
        "steps": [
          {
            "__type__": "SubQAStep",
            "sub_question": "How many different ways are there to insert parentheses into $2\\\\cdot 3\\\\cdot 4\\\\cdot 5+1$?",
            "sub_answer": "The expression $2\\\\cdot 3\\\\cdot 4\\\\cdot 5+1$ can be inserted with 2 parentheses, and each parenthesis can be inserted into 3 different places, so there are 2 * 3 * 3 = 18 different ways to insert parentheses into $2\\\\cdot 3\\\\cdot 4\\\\cdot 5+1$. The answer is 18.",
            "confidence": 0.3333333333333333
          },
          {
            "__type__": "SubQAStep",
            "sub_question": "Now we can answer the question...",
            "sub_answer": "We can obtain 18 different values by inserting parentheses. The answer is 18.",
            "confidence": 0.6666666666666666
          }
        ]
      },
      "action": "Now we can answer the question...",
      "is_terminal": true,
      "fast_reward": 0.6328112942101983,
      "cum_rewards": [0.6495184340264195]
    },
    {
      "id": 7,
      "state": {
        "__type__": "TrajectoryState",
        "steps": [
          {
            "__type__": "SubQAStep",
            "sub_question": "How many ways can we insert parentheses into this expression?",
            "sub_answer": "There are 3 ways to insert parentheses into the expression: we can insert parentheses around 3, 4, 5, or 1. The answer is 3.",
            "confidence": 0.6666666666666666
          },
          {
            "__type__": "SubQAStep",
            "sub_question": "Now we can answer the question...",
            "sub_answer": "There are 3 ways to insert parentheses into the expression: we can insert parentheses around 3, 4, 5, or 1. The answer is 3.",
            "confidence": 1.0
          }
        ]
      },
      "action": "Now we can answer the question...",
      "is_terminal": true,
      "fast_reward": 0.6327822076409035,
      "cum_rewards": [0.7954760886669715]
    },
    {
      "id": 8,
      "state": {
        "__type__": "TrajectoryState",
        "steps": [
          {
            "__type__": "SubQAStep",
            "sub_question": "How many ways can we insert parentheses into this expression?",
            "sub_answer": "There are 3 ways to insert parentheses into the expression: we can insert parentheses around 3, 4, 5, or 1. The answer is 3.",
            "confidence": 0.6666666666666666
          },
          {
            "__type__": "SubQAStep",
            "sub_question": "Now we can answer the question...",
            "sub_answer": "There are 3 ways to insert parentheses into the expression, and the possible values are 121, 144, 180, and 225. In total, there are 3 * 4 = 12 values that can be obtained from the expression. The answer is 12.",
            "confidence": 0.3333333333333333
          }
        ]
      },
      "action": "Now we can answer the question...",
      "is_terminal": true,
      "fast_reward": 0.6327817555077784,
      "cum_rewards": []
    },
    {
      "id": 10,
      "state": {
        "__type__": "TrajectoryState",
        "steps": [
          {
            "__type__": "SubQAStep",
            "sub_question": "How many different ways are there to insert parentheses into the expression?",
            "sub_answer": "The expression $2\\\\cdot 3\\\\cdot 4 \\\\cdot 5 + 1$ has 7 terms, and we can insert parentheses into it in $2^7-1 = 127$ different ways. The answer is 127.",
            "confidence": 0.3333333333333333
          },
          {
            "__type__": "SubQAStep",
            "sub_question": "Now we can answer the question...",
            "sub_answer": "There are 127 different ways to insert parentheses into the expression, and each of them gives a different value, so in total there are 127 different values we can obtain from the expression. The answer is 127.",
            "confidence": 1.0
          }
        ]
      },
      "action": "Now we can answer the question...",
      "is_terminal": true,
      "fast_reward": 0.6327687942070835,
      "cum_rewards": [0.7954676575493711]
    }
  ],
  "query": "The expression $2\\\\cdot 3 \\\\cdot 4\\\\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing (2*(3*4))*(5+1) = 144. In total, how many values can be obtained from the expression $2\\\\cdot 3\\\\cdot 4 \\\\cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).",
  "query_idx": 0
}
'''


def load_terminal_nodes_from_json_str(json_str: str):
    """
    Load terminal nodes from JSON string (for testing without file).
    Uses node.py::SearchNode.from_dict or node.py::MCTSNode.from_dict.
    """
    data = json.loads(json_str)
    
    if not data['terminal_nodes']:
        return {
            'terminal_nodes': [],
            'query': data['query'],
            'query_idx': data['query_idx']
        }
    
    # Determine node type from the data
    if 'cum_rewards' in data['terminal_nodes'][0]:
        node_class = MCTSNode
    else:
        node_class = SearchNode
    
    terminal_nodes = [node_class.from_dict(node_dict) for node_dict in data['terminal_nodes']]
    
    return {
        'terminal_nodes': terminal_nodes,
        'query': data['query'],
        'query_idx': data['query_idx']
    }


# =============================================================================
# Test Classes
# =============================================================================

def check_saved_rap_retrieval_inputs():
    """Characterize common.py::extract_answers_from_terminal_nodes input fallback.

    The saved SubQAStep objects have sub_question/sub_answer but no action field.
    The current helper consequently supplies synthetic ThoughtSteps made from
    node.action to the retriever. Capture this without invoking an LLM.
    """
    data = load_terminal_nodes_from_json_str(REAL_TERMINAL_NODES_JSON)
    received = []

    def retrieve(state, query):
        """Record the actual retrieval input and use LiTS's numeric parser."""
        received.append(state[-1].get_answer())
        return retrieve_answer_from_last_step(state[-1])

    votes, *_ = extract_answers_from_terminal_nodes(data['terminal_nodes'], retrieve, data['query'])
    expected = [str(node.action) for node in data['terminal_nodes']]
    if received != expected or dict(votes) != {"": len(expected)}:
        raise RuntimeError(f"RAP retrieval fallback changed: {received!r}, {dict(votes)!r}")
    breakpoint()  # inspect: received, expected, votes; sub_answer is absent from retrieval input
    return {"case": "rap_retrieval_inputs", "expected": expected, "observed": received}


class TestExtractAnswersFromTerminalNodes:
    """Test the extract_answers_from_terminal_nodes function with mock data."""
    
    def test_basic_extraction(self):
        """Test basic terminal node extraction."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        # Check vote counts (two nodes answer "42", one answers "43")
        assert vote_answers["42"] == 2
        assert vote_answers["43"] == 1
    
    def test_vote_counting(self):
        """Test answer vote counting."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        assert vote_answers["42"] == 2
        assert vote_answers["43"] == 1
    
    def test_reward_aggregation(self):
        """Test reward aggregation by answer."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        assert len(answer_reward_d["42"]) == 2
        assert len(answer_reward_d["43"]) == 1
        assert 0.9 in answer_reward_d["42"]
        assert 0.8 in answer_reward_d["42"]
        assert 0.7 in answer_reward_d["43"]
    
    def test_best_node_selection(self):
        """Test that the node with highest reward is selected."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        assert best_node is not None
        assert best_node.fast_reward == 0.9
        assert best_node.id == 1
    
    def test_trace_reconstruction(self):
        """Test path reconstruction from best node to root."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        assert len(trace) == 2
        assert trace[0] == root
        assert trace[1] == best_node
    
    def test_empty_terminals(self):
        """Test with no terminal nodes."""
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=[],
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        assert len(vote_answers) == 0
        assert len(answer_reward_d) == 0
        assert best_node is None
        assert len(trace) == 0


class TestExtractAnswersWithRealData:
    """
    Check loading and deserialization of saved RAP terminal nodes, without inference.

    eval_search.py::load_terminal_nodes_from_file uses the same node deserialization
    to reconstruct MCTSNode and TrajectoryState objects containing SubQASteps.
    """
    
    def test_load_terminal_nodes(self):
        """
        Test that terminal nodes are correctly loaded and deserialized.
        
        eval_search.py process:
        - load_terminal_nodes_from_file() reads JSON and calls MCTSNode.from_dict()
        """
        data = load_terminal_nodes_from_json_str(REAL_TERMINAL_NODES_JSON)
        terminal_nodes = data['terminal_nodes']
        query = data['query']
        query_idx = data['query_idx']
        
        assert len(terminal_nodes) == 4
        assert query_idx == 0
        
        # Check that nodes are MCTSNode (since they have cum_rewards)
        for node in terminal_nodes:
            assert isinstance(node, MCTSNode)
            assert node.is_terminal == True
    
    def test_state_deserialization(self):
        """
        Test that state (TrajectoryState with SubQAStep) is correctly deserialized.
        
        eval_search.py process:
        - MCTSNode.from_dict() deserializes state using type registry
        - SubQAStep is registered via @register_type decorator
        """
        data = load_terminal_nodes_from_json_str(REAL_TERMINAL_NODES_JSON)
        terminal_nodes = data['terminal_nodes']
        
        node = terminal_nodes[0]
        state = node.state
        
        # State should be a list of SubQAStep objects
        assert isinstance(state, list)
        assert len(state) == 2
        
        # Check SubQAStep attributes
        from lits_benchmark.formulations.rap.structures import SubQAStep
        for step in state:
            assert isinstance(step, SubQAStep)
            assert hasattr(step, 'sub_question')
            assert hasattr(step, 'sub_answer')
            assert hasattr(step, 'confidence')
    
# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestExtractAnswersFromTerminalNodes,
        TestExtractAnswersWithRealData,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {method_name}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                failed_tests.append(f"{test_class.__name__}.{method_name}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print('='*60)
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
