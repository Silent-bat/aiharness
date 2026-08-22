import json
import tempfile
import unittest
from pathlib import Path

from context_compiler.mcp_server import McpServer
from context_compiler.retrieval import classify_task
from context_compiler.service import ContextService


PAYMENTS = """\
class PaymentGateway:
    def capture(self, amount: int) -> str:
        if amount <= 0:
            raise ValueError("amount must be positive")
        return "paid"


def validate_amount(amount: int) -> None:
    if amount <= 0:
        raise ValueError("invalid amount")
"""


CHECKOUT = """\
from payments import PaymentGateway, validate_amount


def checkout_total(items):
    return sum(item["price"] for item in items)


def submit_order(items, gateway: PaymentGateway):
    total = checkout_total(items)
    validate_amount(total)
    return gateway.capture(total)
"""


TEST_CHECKOUT = """\
from checkout import submit_order
from payments import PaymentGateway


def test_submit_order_rejects_empty_cart():
    try:
        submit_order([], PaymentGateway())
    except ValueError:
        return
    raise AssertionError("expected ValueError")
"""


class SystemTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "tests").mkdir()
        (self.root / "payments.py").write_text(PAYMENTS, encoding="utf-8")
        (self.root / "checkout.py").write_text(CHECKOUT, encoding="utf-8")
        (self.root / "tests" / "test_checkout.py").write_text(
            TEST_CHECKOUT, encoding="utf-8"
        )
        self.service = ContextService(self.root)

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def test_incremental_index_and_graph_resolution(self):
        first = self.service.index()
        self.assertEqual(first["scanned"], 3)
        self.assertEqual(first["indexed"], 3)
        self.assertGreater(first["resolved_edges"], 0)

        second = self.service.index()
        self.assertEqual(second["indexed"], 0)
        self.assertEqual(second["unchanged"], 3)

        stats = self.service.stats()
        self.assertGreaterEqual(stats["nodes"], 9)
        self.assertIn("resolved_calls", stats["edge_kinds"])

    def test_task_aware_search_finds_callers_and_tests(self):
        self.service.index()
        results = self.service.search(
            "fix submit_order failing for an empty cart", task_type="debug", limit=12
        )
        names = {result["name"] for result in results}
        self.assertIn("submit_order", names)
        self.assertIn("validate_amount", names)
        self.assertIn("test_submit_order_rejects_empty_cart", names)

    def test_context_compiler_respects_budget(self):
        self.service.index()
        result = self.service.compile_context(
            "debug empty cart submit_order ValueError",
            token_budget=700,
            graph_depth=2,
        )
        self.assertEqual(result["task_type"], "debug")
        self.assertLessEqual(result["estimated_tokens"], 760)
        self.assertIn("submit_order", result["context"])
        self.assertGreater(result["item_count"], 0)

    def test_context_compiler_avoids_redundant_file_after_symbol(self):
        self.service.index()
        result = self.service.compile_context(
            "debug submit_order empty cart", token_budget=900, graph_depth=2
        )
        file_paths = {item["path"] for item in result["items"] if item["key"].startswith("file:")}
        symbol_paths = {item["path"] for item in result["items"] if not item["key"].startswith("file:")}
        self.assertFalse(file_paths.intersection(symbol_paths))

    def test_impact_analysis_returns_related_symbols(self):
        self.service.index()
        result = self.service.impact("validate_amount", depth=2)
        signatures = " ".join(item["signature"] for item in result["impacted"])
        self.assertIn("submit_order", signatures)

    def test_mcp_tools_are_callable(self):
        server = McpServer(self.service)
        initialized = server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        self.assertEqual(initialized["result"]["serverInfo"]["name"], "graph-context-compiler")
        indexed = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "index_repository", "arguments": {}},
            }
        )
        self.assertFalse(indexed["result"]["isError"])
        searched = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search_code_graph",
                    "arguments": {"query": "submit order"},
                },
            }
        )
        self.assertTrue(searched["result"]["structuredContent"]["results"])


class ClassificationTest(unittest.TestCase):
    def test_classifies_common_task_types(self):
        self.assertEqual(classify_task("fix a failing payment test"), "debug")
        self.assertEqual(classify_task("refactor and rename this API"), "refactor")
        self.assertEqual(classify_task("optimize slow database latency"), "performance")


if __name__ == "__main__":
    unittest.main()
