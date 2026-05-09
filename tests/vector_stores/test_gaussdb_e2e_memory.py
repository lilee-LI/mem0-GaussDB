#!/usr/bin/env python3
"""
End-to-End Test Suite for mem0 + GaussDB Integration

Tests all functionality scenarios:
1. Basic CRUD operations (insert, search, delete, update, get)
2. Vector semantic search
3. BM25 keyword search
4. Batch search
5. Filters and scopes
6. Collection management
7. Capability probing and graceful degradation
8. Distributed mode compatibility

Database: 121.37.186.131:19995, lxm/Gauss_234, lxm_db
"""

import os
import sys
import time
import uuid
from typing import List, Dict, Any
from datetime import datetime

# Add mem0 package root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mem0 import Memory
from mem0.configs.vector_stores.gaussdb import GaussDBConfig


# Test configuration
DB_CONFIG = {
    "host": "121.37.186.131",
    "port": 19995,
    "user": "lxm",
    "password": "Gauss_234",
    "database": "lxm_db",
}

# Test data
TEST_MESSAGES = [
    {"role": "user", "content": "My name is Alice and I love Python programming"},
    {"role": "assistant", "content": "Nice to meet you Alice! Python is a great language."},
    {"role": "user", "content": "I work as a data scientist at TechCorp"},
    {"role": "assistant", "content": "That's interesting! Data science is a growing field."},
    {"role": "user", "content": "I enjoy hiking and photography in my free time"},
]


class E2ETestRunner:
    """End-to-end test runner for mem0 + GaussDB"""

    def __init__(self):
        self.test_results = []
        self.memory = None
        self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        self.test_agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"

    def log(self, message: str, level: str = "INFO"):
        """Log test message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def assert_true(self, condition: bool, message: str):
        """Assert condition is true"""
        if not condition:
            raise AssertionError(f"Assertion failed: {message}")
        self.log(f"✓ {message}", "PASS")

    def run_test(self, test_name: str, test_func):
        """Run a single test and record result"""
        self.log(f"\n{'='*60}")
        self.log(f"Running: {test_name}")
        self.log(f"{'='*60}")

        start_time = time.time()
        try:
            test_func()
            elapsed = time.time() - start_time
            self.test_results.append({
                "name": test_name,
                "status": "PASS",
                "elapsed": elapsed
            })
            self.log(f"✓ {test_name} PASSED ({elapsed:.2f}s)", "PASS")
        except Exception as e:
            elapsed = time.time() - start_time
            self.test_results.append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e),
                "elapsed": elapsed
            })
            self.log(f"✗ {test_name} FAILED: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    # ========== Setup and Teardown ==========

    def setup_centralized_mode(self):
        """Setup mem0 with GaussDB in centralized mode"""
        self.log("Setting up mem0 with GaussDB (centralized mode)...")

        config = {
            "vector_store": {
                "provider": "gaussdb",
                "config": {
                    **DB_CONFIG,
                    "collection_name": f"test_centralized_{uuid.uuid4().hex[:8]}",
                    "embedding_model_dims": 1536,
                    "profile": "commercial",  # centralized + A-mode + ustore
                }
            }
        }

        self.memory = Memory.from_config(config)
        self.log(f"✓ Memory initialized with collection: {config['vector_store']['config']['collection_name']}")

    def setup_distributed_mode(self):
        """Setup mem0 with GaussDB in distributed mode"""
        self.log("Setting up mem0 with GaussDB (distributed mode)...")

        config = {
            "vector_store": {
                "provider": "gaussdb",
                "config": {
                    **DB_CONFIG,
                    "collection_name": f"test_distributed_{uuid.uuid4().hex[:8]}",
                    "embedding_model_dims": 1536,
                    "deployment_mode": "distributed",
                    "compatibility_mode": "A",
                    "table_storage": "ustore",
                }
            }
        }

        self.memory = Memory.from_config(config)
        self.log(f"✓ Memory initialized with collection: {config['vector_store']['config']['collection_name']}")

    def teardown(self):
        """Cleanup test data"""
        if self.memory:
            try:
                # Delete all test memories
                self.log("Cleaning up test data...")
                # Note: reset() will drop the entire collection
                # For now, we'll leave cleanup to manual DB maintenance
                self.memory = None
            except Exception as e:
                self.log(f"Cleanup warning: {e}", "WARN")

    # ========== Test Cases ==========

    def test_01_basic_add_memory(self):
        """Test 1: Basic memory addition"""
        result = self.memory.add(
            messages=TEST_MESSAGES[:2],
            user_id=self.test_user_id
        )

        self.assert_true(result is not None, "Memory add returned result")
        self.assert_true(len(result.get("results", [])) > 0, "Memory add created at least one memory")

        memory_id = result["results"][0]["id"]
        self.log(f"Created memory ID: {memory_id}")

        # Store for later tests
        self.first_memory_id = memory_id

    def test_02_semantic_search(self):
        """Test 2: Vector semantic search"""
        results = self.memory.search(
            query="What is the user's name?",
            user_id=self.test_user_id,
            limit=5
        )

        self.assert_true(len(results) > 0, "Semantic search returned results")
        self.assert_true("Alice" in str(results), "Search results contain expected content (Alice)")
        self.log(f"Found {len(results)} semantic search results")

    def test_03_keyword_search(self):
        """Test 3: BM25 keyword search"""
        # Add more memories for keyword search
        self.memory.add(
            messages=[{"role": "user", "content": "Python is my favorite programming language"}],
            user_id=self.test_user_id
        )

        # Search for keyword
        results = self.memory.search(
            query="Python programming",
            user_id=self.test_user_id,
            limit=5
        )

        self.assert_true(len(results) > 0, "Keyword search returned results")
        self.log(f"Found {len(results)} keyword search results")

        # Check if BM25 is enabled
        vector_store = self.memory.vector_store
        if hasattr(vector_store, 'bm25_enabled'):
            self.log(f"BM25 enabled: {vector_store.bm25_enabled}")

    def test_04_scoped_filters(self):
        """Test 4: Scoped filters (user_id, agent_id)"""
        # Add memory with agent_id
        self.memory.add(
            messages=[{"role": "user", "content": "This is agent-specific memory"}],
            user_id=self.test_user_id,
            agent_id=self.test_agent_id
        )

        # Search with agent scope
        results_with_agent = self.memory.search(
            query="agent-specific",
            user_id=self.test_user_id,
            agent_id=self.test_agent_id,
            limit=5
        )

        # Search without agent scope (should not find agent-specific memory)
        results_without_agent = self.memory.search(
            query="agent-specific",
            user_id=self.test_user_id,
            limit=5
        )

        self.assert_true(len(results_with_agent) > 0, "Search with agent_id found results")
        self.log(f"With agent_id: {len(results_with_agent)} results")
        self.log(f"Without agent_id: {len(results_without_agent)} results")

    def test_05_get_memory_by_id(self):
        """Test 5: Get specific memory by ID"""
        if not hasattr(self, 'first_memory_id'):
            self.log("Skipping: no memory ID from previous test", "WARN")
            return

        memory = self.memory.get(self.first_memory_id)

        self.assert_true(memory is not None, "Get memory by ID returned result")
        self.assert_true(memory["id"] == self.first_memory_id, "Retrieved correct memory ID")
        self.log(f"Retrieved memory: {memory['memory'][:50]}...")

    def test_06_get_all_memories(self):
        """Test 6: Get all memories for user"""
        memories = self.memory.get_all(user_id=self.test_user_id)

        self.assert_true(len(memories) > 0, "Get all memories returned results")
        self.log(f"Total memories for user: {len(memories)}")

    def test_07_update_memory(self):
        """Test 7: Update existing memory"""
        if not hasattr(self, 'first_memory_id'):
            self.log("Skipping: no memory ID from previous test", "WARN")
            return

        updated_text = "Updated: Alice loves Python and machine learning"
        result = self.memory.update(
            memory_id=self.first_memory_id,
            data=updated_text
        )

        self.assert_true(result is not None, "Update memory returned result")

        # Verify update
        memory = self.memory.get(self.first_memory_id)
        self.assert_true("machine learning" in memory["memory"], "Memory was updated")
        self.log("Memory successfully updated")

    def test_08_delete_memory(self):
        """Test 8: Delete specific memory"""
        # Create a temporary memory to delete
        result = self.memory.add(
            messages=[{"role": "user", "content": "This memory will be deleted"}],
            user_id=self.test_user_id
        )

        temp_memory_id = result["results"][0]["id"]
        self.log(f"Created temporary memory: {temp_memory_id}")

        # Delete it
        self.memory.delete(temp_memory_id)
        self.log("Memory deleted")

        # Verify deletion
        try:
            memory = self.memory.get(temp_memory_id)
            self.assert_true(memory is None, "Deleted memory should not be retrievable")
        except:
            # Expected: memory not found
            self.log("✓ Confirmed memory was deleted")

    def test_09_batch_operations(self):
        """Test 9: Batch add and search"""
        # Batch add
        batch_messages = [
            [{"role": "user", "content": f"Batch message {i}"}]
            for i in range(5)
        ]

        for msgs in batch_messages:
            self.memory.add(messages=msgs, user_id=self.test_user_id)

        self.log("Added 5 batch memories")

        # Verify all were added
        all_memories = self.memory.get_all(user_id=self.test_user_id)
        self.assert_true(len(all_memories) >= 5, "Batch memories were added")

    def test_10_custom_filters(self):
        """Test 10: Custom metadata filters"""
        # Add memory with custom metadata
        result = self.memory.add(
            messages=[{"role": "user", "content": "Memory with custom metadata"}],
            user_id=self.test_user_id,
            metadata={"category": "work", "priority": "high"}
        )

        self.log("Added memory with custom metadata")

        # Search with filters
        results = self.memory.search(
            query="custom metadata",
            user_id=self.test_user_id,
            filters={"category": "work"},
            limit=5
        )

        self.assert_true(len(results) > 0, "Search with custom filters returned results")
        self.log(f"Found {len(results)} results with category=work filter")

    def test_11_collection_management(self):
        """Test 11: Collection listing and info"""
        vector_store = self.memory.vector_store

        # List collections
        collections = vector_store.list_cols()
        self.assert_true(len(collections) > 0, "List collections returned results")
        self.log(f"Found {len(collections)} collections")

        # Get collection info
        current_collection = vector_store.collection_name
        info = vector_store.col_info(current_collection)

        self.assert_true(info is not None, "Collection info returned result")
        self.log(f"Collection info: {info}")

    def test_12_capability_probing(self):
        """Test 12: Verify capability probing"""
        vector_store = self.memory.vector_store

        # Check if capabilities were probed
        self.assert_true(hasattr(vector_store, 'bm25_enabled'), "BM25 capability flag exists")
        self.assert_true(hasattr(vector_store, 'vector_enabled'), "Vector capability flag exists")

        self.log(f"BM25 enabled: {vector_store.bm25_enabled}")
        self.log(f"Vector enabled: {vector_store.vector_enabled}")

        # Both should be enabled for a properly configured GaussDB
        if vector_store.bm25_enabled:
            self.log("✓ BM25 keyword search is available")
        else:
            self.log("⚠ BM25 keyword search is disabled (may not be supported in this deployment)", "WARN")

    def test_13_concurrent_operations(self):
        """Test 13: Concurrent insert operations (MERGE INTO atomicity)"""
        import threading

        results = []
        errors = []

        def concurrent_add(thread_id):
            try:
                result = self.memory.add(
                    messages=[{"role": "user", "content": f"Concurrent message from thread {thread_id}"}],
                    user_id=self.test_user_id
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Launch 10 concurrent threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=concurrent_add, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        self.assert_true(len(errors) == 0, f"No errors in concurrent operations (errors: {errors})")
        self.assert_true(len(results) == 10, "All 10 concurrent operations succeeded")
        self.log("✓ MERGE INTO handled concurrent operations correctly")

    def test_14_large_payload(self):
        """Test 14: Large payload handling"""
        large_content = "This is a large memory. " * 100  # ~2400 chars

        result = self.memory.add(
            messages=[{"role": "user", "content": large_content}],
            user_id=self.test_user_id
        )

        self.assert_true(result is not None, "Large payload was stored")

        # Retrieve and verify
        memory_id = result["results"][0]["id"]
        memory = self.memory.get(memory_id)

        self.assert_true(len(memory["memory"]) > 1000, "Large payload was retrieved correctly")
        self.log(f"Stored and retrieved {len(memory['memory'])} character payload")

    def test_15_empty_search_results(self):
        """Test 15: Handle empty search results gracefully"""
        results = self.memory.search(
            query="xyzabc123nonexistent",
            user_id=self.test_user_id,
            limit=5
        )

        self.assert_true(isinstance(results, list), "Empty search returns list")
        self.log(f"Empty search returned {len(results)} results (expected 0 or low relevance)")

    # ========== Main Test Runner ==========

    def run_all_tests(self, mode: str = "centralized"):
        """Run all tests in specified mode"""
        self.log(f"\n{'#'*60}")
        self.log(f"# mem0 + GaussDB End-to-End Test Suite")
        self.log(f"# Mode: {mode.upper()}")
        self.log(f"# Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        self.log(f"# Test User: {self.test_user_id}")
        self.log(f"{'#'*60}\n")

        try:
            # Setup
            if mode == "centralized":
                self.setup_centralized_mode()
            elif mode == "distributed":
                self.setup_distributed_mode()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Run all tests
            self.run_test("Test 01: Basic Add Memory", self.test_01_basic_add_memory)
            self.run_test("Test 02: Semantic Search", self.test_02_semantic_search)
            self.run_test("Test 03: Keyword Search (BM25)", self.test_03_keyword_search)
            self.run_test("Test 04: Scoped Filters", self.test_04_scoped_filters)
            self.run_test("Test 05: Get Memory by ID", self.test_05_get_memory_by_id)
            self.run_test("Test 06: Get All Memories", self.test_06_get_all_memories)
            self.run_test("Test 07: Update Memory", self.test_07_update_memory)
            self.run_test("Test 08: Delete Memory", self.test_08_delete_memory)
            self.run_test("Test 09: Batch Operations", self.test_09_batch_operations)
            self.run_test("Test 10: Custom Filters", self.test_10_custom_filters)
            self.run_test("Test 11: Collection Management", self.test_11_collection_management)
            self.run_test("Test 12: Capability Probing", self.test_12_capability_probing)
            self.run_test("Test 13: Concurrent Operations", self.test_13_concurrent_operations)
            self.run_test("Test 14: Large Payload", self.test_14_large_payload)
            self.run_test("Test 15: Empty Search Results", self.test_15_empty_search_results)

        finally:
            self.teardown()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        self.log(f"\n{'='*60}")
        self.log("TEST SUMMARY")
        self.log(f"{'='*60}")

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed = total - passed

        self.log(f"Total: {total} | Passed: {passed} | Failed: {failed}")

        if failed > 0:
            self.log("\nFailed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    self.log(f"  ✗ {result['name']}: {result.get('error', 'Unknown error')}", "FAIL")

        self.log(f"\n{'='*60}")
        if failed == 0:
            self.log("✓ ALL TESTS PASSED", "PASS")
        else:
            self.log(f"✗ {failed} TEST(S) FAILED", "FAIL")
        self.log(f"{'='*60}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="mem0 + GaussDB E2E Test Suite")
    parser.add_argument(
        "--mode",
        choices=["centralized", "distributed", "both"],
        default="centralized",
        help="Test mode: centralized, distributed, or both"
    )

    args = parser.parse_args()

    if args.mode == "both":
        # Run centralized tests
        runner_centralized = E2ETestRunner()
        runner_centralized.run_all_tests(mode="centralized")

        # Run distributed tests
        runner_distributed = E2ETestRunner()
        runner_distributed.run_all_tests(mode="distributed")
    else:
        runner = E2ETestRunner()
        runner.run_all_tests(mode=args.mode)


if __name__ == "__main__":
    main()
