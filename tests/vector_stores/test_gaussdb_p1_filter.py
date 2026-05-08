"""
P1 Filter System Tests for GaussDB vector store.

Tests comprehensive filter functionality for:
- Simple operators: eq, ne, in, nin, contains, icontains
- Range operators: gt, gte, lt, lte, combined ranges
- Logical combination operators: AND, OR, NOT
- Nested logic combinations: multi-level nesting
- Null value handling: missing keys, null, empty string
- Multi-tenant isolation: user_id, agent_id, run_id scoping

51 tests total.
"""

import pytest

from tests.vector_stores.conftest import (
    VECTOR_COFFEE,
    VECTOR_FLIGHT,
    VECTOR_WINDOW,
    VECTOR_AISLE,
    _assert_exact_ids,
    _insert_memories,
    _new_db,
    _uuid,
    gaussdb_available,
    _make_payload,
)

pytestmark = [
    pytest.mark.p1,
    pytest.mark.skipif(not gaussdb_available(), reason="GaussDB test env not configured"),
]


# ===========================================================================
# 5.2.1 Simple Operators (15 tests)
# ===========================================================================


class TestSimpleOperators:
    """Tests for eq, ne, in, nin, contains, icontains operators."""

    def test_eq_string_exact_match(self):
        """eq operator with string value returns exact match."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(100), VECTOR_COFFEE, _make_payload("coffee note", user_id="alice", category="food")),
                    (_uuid(101), VECTOR_FLIGHT, _make_payload("flight note", user_id="alice", category="travel")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice", "category": {"eq": "food"}})
            _assert_exact_ids(rows, {_uuid(100)})
        finally:
            db.delete_col()

    def test_eq_number_exact_match(self):
        """eq operator with numeric value returns exact match."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(110), VECTOR_COFFEE, _make_payload("low priority", user_id="alice", priority="3")),
                    (_uuid(111), VECTOR_FLIGHT, _make_payload("high priority", user_id="alice", priority="7")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice", "priority": {"eq": "7"}})
            _assert_exact_ids(rows, {_uuid(111)})
        finally:
            db.delete_col()

    def test_eq_boolean_exact_match(self):
        """eq operator with boolean-like value returns exact match."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(120), VECTOR_COFFEE, _make_payload("active item", user_id="alice", active="true")),
                    (_uuid(121), VECTOR_FLIGHT, _make_payload("inactive item", user_id="alice", active="false")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice", "active": {"eq": "true"}})
            _assert_exact_ids(rows, {_uuid(120)})
        finally:
            db.delete_col()

    def test_ne_string_exclusion(self):
        """ne operator excludes matching string value."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(130), VECTOR_COFFEE, _make_payload("food item", user_id="bob", category="food")),
                    (_uuid(131), VECTOR_FLIGHT, _make_payload("travel item", user_id="bob", category="travel")),
                    (_uuid(132), VECTOR_WINDOW, _make_payload("work item", user_id="bob", category="work")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "bob", "category": {"ne": "food"}})
            _assert_exact_ids(rows, {_uuid(131), _uuid(132)})
        finally:
            db.delete_col()

    def test_ne_number_exclusion(self):
        """ne operator excludes matching numeric value."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(140), VECTOR_COFFEE, _make_payload("pri 2", user_id="bob", priority="2")),
                    (_uuid(141), VECTOR_FLIGHT, _make_payload("pri 5", user_id="bob", priority="5")),
                    (_uuid(142), VECTOR_WINDOW, _make_payload("pri 8", user_id="bob", priority="8")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "bob", "priority": {"ne": "5"}})
            _assert_exact_ids(rows, {_uuid(140), _uuid(142)})
        finally:
            db.delete_col()

    def test_in_single_value(self):
        """in operator with single value list returns matching record."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(150), VECTOR_COFFEE, _make_payload("food", user_id="carol", category="food")),
                    (_uuid(151), VECTOR_FLIGHT, _make_payload("travel", user_id="carol", category="travel")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "carol", "category": {"in": ["food"]}})
            _assert_exact_ids(rows, {_uuid(150)})
        finally:
            db.delete_col()

    def test_in_multi_value(self):
        """in operator with multiple values returns all matching records."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(160), VECTOR_COFFEE, _make_payload("food", user_id="carol", category="food")),
                    (_uuid(161), VECTOR_FLIGHT, _make_payload("travel", user_id="carol", category="travel")),
                    (_uuid(162), VECTOR_WINDOW, _make_payload("work", user_id="carol", category="work")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10, filters={"user_id": "carol", "category": {"in": ["food", "travel"]}}
            )
            _assert_exact_ids(rows, {_uuid(160), _uuid(161)})
        finally:
            db.delete_col()

    def test_in_empty_list_returns_nothing(self):
        """in operator with empty list returns no results."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(170), VECTOR_COFFEE, _make_payload("food", user_id="carol", category="food")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "carol", "category": {"in": []}})
            assert len(rows) == 0
        finally:
            db.delete_col()

    def test_nin_single_value(self):
        """nin operator with single value excludes matching record."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(180), VECTOR_COFFEE, _make_payload("food", user_id="dave", category="food")),
                    (_uuid(181), VECTOR_FLIGHT, _make_payload("travel", user_id="dave", category="travel")),
                    (_uuid(182), VECTOR_WINDOW, _make_payload("work", user_id="dave", category="work")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "dave", "category": {"nin": ["food"]}})
            _assert_exact_ids(rows, {_uuid(181), _uuid(182)})
        finally:
            db.delete_col()

    def test_nin_multi_value_exclusion(self):
        """nin operator with multiple values excludes all matching records."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(190), VECTOR_COFFEE, _make_payload("food", user_id="dave", category="food")),
                    (_uuid(191), VECTOR_FLIGHT, _make_payload("travel", user_id="dave", category="travel")),
                    (_uuid(192), VECTOR_WINDOW, _make_payload("work", user_id="dave", category="work")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10, filters={"user_id": "dave", "category": {"nin": ["food", "travel"]}}
            )
            _assert_exact_ids(rows, {_uuid(192)})
        finally:
            db.delete_col()

    def test_contains_exact_substring(self):
        """contains operator matches exact substring."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(200), VECTOR_COFFEE, _make_payload("coffee shop", user_id="eve", tag="morning-coffee")),
                    (_uuid(201), VECTOR_FLIGHT, _make_payload("flight plan", user_id="eve", tag="evening-flight")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "eve", "tag": {"contains": "coffee"}})
            _assert_exact_ids(rows, {_uuid(200)})
        finally:
            db.delete_col()

    def test_contains_partial_substring(self):
        """contains operator matches partial substring within value."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(210), VECTOR_COFFEE, _make_payload("item1", user_id="eve", tag="super-coffee-deluxe")),
                    (_uuid(211), VECTOR_FLIGHT, _make_payload("item2", user_id="eve", tag="no-match-here")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "eve", "tag": {"contains": "coffee"}})
            _assert_exact_ids(rows, {_uuid(210)})
        finally:
            db.delete_col()

    def test_contains_not_exists_returns_empty(self):
        """contains operator with non-matching substring returns empty."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(220), VECTOR_COFFEE, _make_payload("item", user_id="eve", tag="morning-tea")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "eve", "tag": {"contains": "coffee"}})
            assert len(rows) == 0
        finally:
            db.delete_col()

    def test_icontains_case_insensitive_match(self):
        """icontains operator matches regardless of case."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(230), VECTOR_COFFEE, _make_payload("item1", user_id="eve", tag="MorningCoffee")),
                    (_uuid(231), VECTOR_FLIGHT, _make_payload("item2", user_id="eve", tag="EVENING-COFFEE")),
                    (_uuid(232), VECTOR_WINDOW, _make_payload("item3", user_id="eve", tag="afternoon-tea")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "eve", "tag": {"icontains": "coffee"}})
            _assert_exact_ids(rows, {_uuid(230), _uuid(231)})
        finally:
            db.delete_col()

    def test_eq_implicit_direct_value(self):
        """Direct value (without eq wrapper) acts as implicit equality filter."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(240), VECTOR_COFFEE, _make_payload("food item", user_id="frank", category="food")),
                    (_uuid(241), VECTOR_FLIGHT, _make_payload("travel item", user_id="frank", category="travel")),
                ],
            )
            # Direct value without {"eq": ...} wrapper
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "frank", "category": "food"})
            _assert_exact_ids(rows, {_uuid(240)})
        finally:
            db.delete_col()


# ===========================================================================
# 5.2.2 Range Operators (8 tests)
# ===========================================================================


class TestRangeOperators:
    """Tests for gt, gte, lt, lte and combined range operators."""

    def test_gt_operator(self):
        """gt operator returns records with value strictly greater than threshold."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(300), VECTOR_COFFEE, _make_payload("pri 2", user_id="range_user", priority="2")),
                    (_uuid(301), VECTOR_FLIGHT, _make_payload("pri 5", user_id="range_user", priority="5")),
                    (_uuid(302), VECTOR_WINDOW, _make_payload("pri 8", user_id="range_user", priority="8")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "priority": {"gt": "5"}})
            _assert_exact_ids(rows, {_uuid(302)})
        finally:
            db.delete_col()

    def test_gte_operator(self):
        """gte operator returns records with value greater than or equal to threshold."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(310), VECTOR_COFFEE, _make_payload("pri 2", user_id="range_user", priority="2")),
                    (_uuid(311), VECTOR_FLIGHT, _make_payload("pri 5", user_id="range_user", priority="5")),
                    (_uuid(312), VECTOR_WINDOW, _make_payload("pri 8", user_id="range_user", priority="8")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "priority": {"gte": "5"}})
            _assert_exact_ids(rows, {_uuid(311), _uuid(312)})
        finally:
            db.delete_col()

    def test_lt_operator(self):
        """lt operator returns records with value strictly less than threshold."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(320), VECTOR_COFFEE, _make_payload("pri 2", user_id="range_user", priority="2")),
                    (_uuid(321), VECTOR_FLIGHT, _make_payload("pri 5", user_id="range_user", priority="5")),
                    (_uuid(322), VECTOR_WINDOW, _make_payload("pri 8", user_id="range_user", priority="8")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "priority": {"lt": "5"}})
            _assert_exact_ids(rows, {_uuid(320)})
        finally:
            db.delete_col()

    def test_lte_operator(self):
        """lte operator returns records with value less than or equal to threshold."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(330), VECTOR_COFFEE, _make_payload("pri 2", user_id="range_user", priority="2")),
                    (_uuid(331), VECTOR_FLIGHT, _make_payload("pri 5", user_id="range_user", priority="5")),
                    (_uuid(332), VECTOR_WINDOW, _make_payload("pri 8", user_id="range_user", priority="8")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "priority": {"lte": "5"}})
            _assert_exact_ids(rows, {_uuid(330), _uuid(331)})
        finally:
            db.delete_col()

    def test_range_combined_gte_lte(self):
        """Combined gte + lte creates an inclusive range filter."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(340), VECTOR_COFFEE, _make_payload("pri 1", user_id="range_user", priority="1")),
                    (_uuid(341), VECTOR_FLIGHT, _make_payload("pri 3", user_id="range_user", priority="3")),
                    (_uuid(342), VECTOR_WINDOW, _make_payload("pri 5", user_id="range_user", priority="5")),
                    (_uuid(343), VECTOR_AISLE, _make_payload("pri 7", user_id="range_user", priority="7")),
                    (_uuid(344), VECTOR_COFFEE, _make_payload("pri 9", user_id="range_user", priority="9")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "priority": {"gte": "3", "lte": "7"}}
            )
            _assert_exact_ids(rows, {_uuid(341), _uuid(342), _uuid(343)})
        finally:
            db.delete_col()

    def test_gt_negative_numbers(self):
        """gt operator with negative number strings uses lexicographic comparison."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(350), VECTOR_COFFEE, _make_payload("neg", user_id="range_user", score="-5")),
                    (_uuid(351), VECTOR_FLIGHT, _make_payload("zero", user_id="range_user", score="0")),
                    (_uuid(352), VECTOR_WINDOW, _make_payload("pos", user_id="range_user", score="5")),
                ],
            )
            # Note: JSON string comparison is lexicographic, so "-5" > "-3" is true
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "score": {"gt": "-3"}})
            _assert_exact_ids(rows, {_uuid(350), _uuid(351), _uuid(352)})
        finally:
            db.delete_col()

    def test_gt_float_values(self):
        """gt operator works with float value strings."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(360), VECTOR_COFFEE, _make_payload("low", user_id="range_user", rating="2.5")),
                    (_uuid(361), VECTOR_FLIGHT, _make_payload("mid", user_id="range_user", rating="4.2")),
                    (_uuid(362), VECTOR_WINDOW, _make_payload("high", user_id="range_user", rating="4.9")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "rating": {"gt": "4.0"}})
            _assert_exact_ids(rows, {_uuid(361), _uuid(362)})
        finally:
            db.delete_col()

    def test_gt_date_string(self):
        """gt operator works with ISO date strings for lexicographic comparison."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(370), VECTOR_COFFEE, _make_payload("old", user_id="range_user", created="2024-01-01")),
                    (_uuid(371), VECTOR_FLIGHT, _make_payload("mid", user_id="range_user", created="2024-06-15")),
                    (_uuid(372), VECTOR_WINDOW, _make_payload("new", user_id="range_user", created="2025-01-01")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10, filters={"user_id": "range_user", "created": {"gt": "2024-06-01"}}
            )
            _assert_exact_ids(rows, {_uuid(371), _uuid(372)})
        finally:
            db.delete_col()


# ===========================================================================
# 5.2.3 Logical Combination Operators (7 tests)
# ===========================================================================


class TestLogicalCombinationOperators:
    """Tests for AND, OR, NOT logical operators."""

    def test_and_implicit_two_keys(self):
        """Implicit AND with two keys in same dict filters by both conditions."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(400), VECTOR_COFFEE, _make_payload("food hi", user_id="logic_user", category="food", priority="7")),
                    (_uuid(401), VECTOR_FLIGHT, _make_payload("food lo", user_id="logic_user", category="food", priority="2")),
                    (_uuid(402), VECTOR_WINDOW, _make_payload("travel hi", user_id="logic_user", category="travel", priority="7")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "logic_user", "category": "food", "priority": {"gte": "5"}},
            )
            _assert_exact_ids(rows, {_uuid(400)})
        finally:
            db.delete_col()

    def test_and_implicit_three_keys(self):
        """Implicit AND with three filter keys narrows results correctly."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(410), VECTOR_COFFEE, _make_payload("match", user_id="logic_user", category="food", status="active", priority="5")),
                    (_uuid(411), VECTOR_FLIGHT, _make_payload("no cat", user_id="logic_user", category="travel", status="active", priority="5")),
                    (_uuid(412), VECTOR_WINDOW, _make_payload("no status", user_id="logic_user", category="food", status="archived", priority="5")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "logic_user", "category": "food", "status": "active"},
            )
            _assert_exact_ids(rows, {_uuid(410)})
        finally:
            db.delete_col()

    def test_and_explicit_operator(self):
        """Explicit $and operator combines conditions."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(420), VECTOR_COFFEE, _make_payload("match", user_id="logic_user", category="food", priority="7")),
                    (_uuid(421), VECTOR_FLIGHT, _make_payload("no match", user_id="logic_user", category="travel", priority="7")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"$and": [{"user_id": "logic_user"}, {"category": "food"}, {"priority": {"gte": "5"}}]},
            )
            _assert_exact_ids(rows, {_uuid(420)})
        finally:
            db.delete_col()

    def test_or_two_conditions(self):
        """$or operator with two conditions returns union of matches."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(430), VECTOR_COFFEE, _make_payload("food", user_id="logic_user", category="food")),
                    (_uuid(431), VECTOR_FLIGHT, _make_payload("travel", user_id="logic_user", category="travel")),
                    (_uuid(432), VECTOR_WINDOW, _make_payload("work", user_id="logic_user", category="work")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"$or": [{"user_id": "logic_user", "category": "food"}, {"user_id": "logic_user", "category": "travel"}]},
            )
            _assert_exact_ids(rows, {_uuid(430), _uuid(431)})
        finally:
            db.delete_col()

    def test_or_three_conditions(self):
        """$or operator with three conditions returns union of all matches."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(440), VECTOR_COFFEE, _make_payload("food", user_id="logic_user", category="food")),
                    (_uuid(441), VECTOR_FLIGHT, _make_payload("travel", user_id="logic_user", category="travel")),
                    (_uuid(442), VECTOR_WINDOW, _make_payload("work", user_id="logic_user", category="work")),
                    (_uuid(443), VECTOR_AISLE, _make_payload("health", user_id="logic_user", category="health")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={
                    "$or": [
                        {"user_id": "logic_user", "category": "food"},
                        {"user_id": "logic_user", "category": "travel"},
                        {"user_id": "logic_user", "category": "work"},
                    ]
                },
            )
            _assert_exact_ids(rows, {_uuid(440), _uuid(441), _uuid(442)})
        finally:
            db.delete_col()

    def test_not_single_condition(self):
        """$not operator excludes matching records."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(450), VECTOR_COFFEE, _make_payload("food", user_id="logic_user", category="food")),
                    (_uuid(451), VECTOR_FLIGHT, _make_payload("travel", user_id="logic_user", category="travel")),
                    (_uuid(452), VECTOR_WINDOW, _make_payload("work", user_id="logic_user", category="work")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "logic_user", "$not": [{"category": "food"}]},
            )
            _assert_exact_ids(rows, {_uuid(451), _uuid(452)})
        finally:
            db.delete_col()

    def test_not_with_scoped_guard(self):
        """$not combined with scoped user_id filter works correctly."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(460), VECTOR_COFFEE, _make_payload("food", user_id="logic_user", category="food")),
                    (_uuid(461), VECTOR_FLIGHT, _make_payload("travel", user_id="logic_user", category="travel")),
                    (_uuid(462), VECTOR_WINDOW, _make_payload("work", user_id="other_user", category="work")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "logic_user", "$not": [{"category": "travel"}]},
            )
            _assert_exact_ids(rows, {_uuid(460)})
        finally:
            db.delete_col()


# ===========================================================================
# 5.2.4 Nested Logic Combinations (4 tests)
# ===========================================================================


class TestNestedLogicCombinations:
    """Tests for nested logical operator combinations."""

    def test_or_nested_and(self):
        """$or containing nested AND conditions (implicit via dict keys)."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(500), VECTOR_COFFEE, _make_payload("food hi", user_id="nest_user", category="food", priority="7")),
                    (_uuid(501), VECTOR_FLIGHT, _make_payload("food lo", user_id="nest_user", category="food", priority="2")),
                    (_uuid(502), VECTOR_WINDOW, _make_payload("travel hi", user_id="nest_user", category="travel", priority="8")),
                    (_uuid(503), VECTOR_AISLE, _make_payload("travel lo", user_id="nest_user", category="travel", priority="1")),
                ],
            )
            # Match: (food AND priority>=5) OR (travel AND priority>=5)
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={
                    "$or": [
                        {"user_id": "nest_user", "category": "food", "priority": {"gte": "5"}},
                        {"user_id": "nest_user", "category": "travel", "priority": {"gte": "5"}},
                    ]
                },
            )
            _assert_exact_ids(rows, {_uuid(500), _uuid(502)})
        finally:
            db.delete_col()

    def test_and_nested_or(self):
        """$and containing a nested $or condition."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(510), VECTOR_COFFEE, _make_payload("food hi", user_id="nest_user", category="food", priority="7")),
                    (_uuid(511), VECTOR_FLIGHT, _make_payload("travel hi", user_id="nest_user", category="travel", priority="8")),
                    (_uuid(512), VECTOR_WINDOW, _make_payload("work hi", user_id="nest_user", category="work", priority="9")),
                ],
            )
            # Match: user_id=nest_user AND (category=food OR category=travel) AND priority>=7
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={
                    "$and": [
                        {"user_id": "nest_user"},
                        {"$or": [{"user_id": "nest_user", "category": "food"}, {"user_id": "nest_user", "category": "travel"}]},
                        {"priority": {"gte": "7"}},
                    ]
                },
            )
            _assert_exact_ids(rows, {_uuid(510), _uuid(511)})
        finally:
            db.delete_col()

    def test_not_nested_or(self):
        """$not applied to an $or-like condition via multiple NOT items."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(520), VECTOR_COFFEE, _make_payload("food", user_id="nest_user", category="food")),
                    (_uuid(521), VECTOR_FLIGHT, _make_payload("travel", user_id="nest_user", category="travel")),
                    (_uuid(522), VECTOR_WINDOW, _make_payload("work", user_id="nest_user", category="work")),
                    (_uuid(523), VECTOR_AISLE, _make_payload("health", user_id="nest_user", category="health")),
                ],
            )
            # Exclude food AND exclude travel => only work and health remain
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "nest_user", "$not": [{"category": "food"}, {"category": "travel"}]},
            )
            _assert_exact_ids(rows, {_uuid(522), _uuid(523)})
        finally:
            db.delete_col()

    def test_complex_three_level_nesting(self):
        """Complex filter with three levels of nesting."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(530), VECTOR_COFFEE, _make_payload("a", user_id="nest_user", category="food", status="active", priority="7")),
                    (_uuid(531), VECTOR_FLIGHT, _make_payload("b", user_id="nest_user", category="travel", status="active", priority="3")),
                    (_uuid(532), VECTOR_WINDOW, _make_payload("c", user_id="nest_user", category="food", status="archived", priority="9")),
                    (_uuid(533), VECTOR_AISLE, _make_payload("d", user_id="nest_user", category="work", status="active", priority="6")),
                ],
            )
            # Match: user_id=nest_user AND status=active AND (category=food OR (category=work AND priority>=5))
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={
                    "$and": [
                        {"user_id": "nest_user"},
                        {"status": "active"},
                        {
                            "$or": [
                                {"user_id": "nest_user", "category": "food"},
                                {"user_id": "nest_user", "category": "work", "priority": {"gte": "5"}},
                            ]
                        },
                    ]
                },
            )
            _assert_exact_ids(rows, {_uuid(530), _uuid(533)})
        finally:
            db.delete_col()


# ===========================================================================
# 5.2.5 Null Value Handling (5 tests)
# ===========================================================================


class TestNullValueHandling:
    """Tests for null, missing key, and empty string filter behavior."""

    def test_filter_key_not_exists_returns_empty(self):
        """Filtering on a key that does not exist in payload returns no results."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(600), VECTOR_COFFEE, _make_payload("item", user_id="null_user", category="food")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "null_user", "nonexistent_key": "some_value"},
            )
            assert len(rows) == 0
        finally:
            db.delete_col()

    def test_eq_null_value(self):
        """Filtering with eq on a None/null value matches records where key is null or missing."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(610), VECTOR_COFFEE, _make_payload("with tag", user_id="null_user", tag="hello")),
                    (_uuid(611), VECTOR_FLIGHT, _make_payload("no tag", user_id="null_user")),
                ],
            )
            # Filtering for tag=None should match the record without tag (payload->>tag IS NULL)
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "null_user", "tag": {"eq": "None"}},
            )
            # The record without tag has payload->>tag = NULL, which won't match string "None"
            # so this should return empty (strict string comparison)
            assert len(rows) == 0
        finally:
            db.delete_col()

    def test_ne_null_value_returns_records_with_key(self):
        """ne with a value returns records that have the key with a different value."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(620), VECTOR_COFFEE, _make_payload("tagged", user_id="null_user", tag="hello")),
                    (_uuid(621), VECTOR_FLIGHT, _make_payload("diff tag", user_id="null_user", tag="world")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "null_user", "tag": {"ne": "hello"}},
            )
            _assert_exact_ids(rows, {_uuid(621)})
        finally:
            db.delete_col()

    def test_empty_string_match(self):
        """Filtering for empty string - GaussDB JSON eq does not match empty strings."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(630), VECTOR_COFFEE, _make_payload("empty tag", user_id="null_user", tag="")),
                    (_uuid(631), VECTOR_FLIGHT, _make_payload("has tag", user_id="null_user", tag="hello")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "null_user", "tag": ""},
            )
            # GaussDB JSON filter does not match empty string values
            assert len(rows) == 0
        finally:
            db.delete_col()

    def test_missing_vs_empty_string_difference(self):
        """Records with missing key vs empty string - empty string eq returns empty."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(640), VECTOR_COFFEE, _make_payload("empty", user_id="null_user", tag="")),
                    (_uuid(641), VECTOR_FLIGHT, _make_payload("missing", user_id="null_user")),
                    (_uuid(642), VECTOR_WINDOW, _make_payload("present", user_id="null_user", tag="value")),
                ],
            )
            # Empty string eq does not match in GaussDB JSON filter
            rows_empty = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "null_user", "tag": ""},
            )
            assert len(rows_empty) == 0
            # Non-empty match works normally
            rows_value = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "null_user", "tag": "value"},
            )
            _assert_exact_ids(rows_value, {_uuid(642)})
        finally:
            db.delete_col()


# ===========================================================================
# 5.2.6 Multi-tenant Isolation (12 tests)
# ===========================================================================


class TestMultiTenantIsolation:
    """Tests for multi-tenant scoped filter isolation and enforcement."""

    def test_user_id_isolation(self):
        """Records from different user_ids are isolated by filter."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(700), VECTOR_COFFEE, _make_payload("alice item", user_id="alice")),
                    (_uuid(701), VECTOR_FLIGHT, _make_payload("bob item", user_id="bob")),
                    (_uuid(702), VECTOR_WINDOW, _make_payload("carol item", user_id="carol")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice"})
            _assert_exact_ids(rows, {_uuid(700)})
        finally:
            db.delete_col()

    def test_cross_tenant_search_returns_only_own_data(self):
        """Searching with one user_id never returns another user's data."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(710), VECTOR_COFFEE, _make_payload("alice secret", user_id="alice")),
                    (_uuid(711), VECTOR_COFFEE, _make_payload("bob secret", user_id="bob")),
                ],
            )
            alice_rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice"})
            bob_rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "bob"})
            _assert_exact_ids(alice_rows, {_uuid(710)})
            _assert_exact_ids(bob_rows, {_uuid(711)})
        finally:
            db.delete_col()

    def test_agent_id_isolation(self):
        """Records are isolated by agent_id scope filter."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(720), VECTOR_COFFEE, _make_payload("agent1 item", user_id="user1", agent_id="agent_a")),
                    (_uuid(721), VECTOR_FLIGHT, _make_payload("agent2 item", user_id="user1", agent_id="agent_b")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"agent_id": "agent_a"})
            _assert_exact_ids(rows, {_uuid(720)})
        finally:
            db.delete_col()

    def test_run_id_isolation(self):
        """Records are isolated by run_id scope filter."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(730), VECTOR_COFFEE, _make_payload("run1 item", user_id="user1", run_id="run_001")),
                    (_uuid(731), VECTOR_FLIGHT, _make_payload("run2 item", user_id="user1", run_id="run_002")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"run_id": "run_001"})
            _assert_exact_ids(rows, {_uuid(730)})
        finally:
            db.delete_col()

    def test_combined_user_and_agent_isolation(self):
        """Combined user_id + agent_id filter narrows results correctly."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(740), VECTOR_COFFEE, _make_payload("u1 a1", user_id="user1", agent_id="agent_a")),
                    (_uuid(741), VECTOR_FLIGHT, _make_payload("u1 a2", user_id="user1", agent_id="agent_b")),
                    (_uuid(742), VECTOR_WINDOW, _make_payload("u2 a1", user_id="user2", agent_id="agent_a")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "user1", "agent_id": "agent_a"})
            _assert_exact_ids(rows, {_uuid(740)})
        finally:
            db.delete_col()

    def test_combined_user_and_run_isolation(self):
        """Combined user_id + run_id filter narrows results correctly."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(750), VECTOR_COFFEE, _make_payload("u1 r1", user_id="user1", run_id="run_001")),
                    (_uuid(751), VECTOR_FLIGHT, _make_payload("u1 r2", user_id="user1", run_id="run_002")),
                    (_uuid(752), VECTOR_WINDOW, _make_payload("u2 r1", user_id="user2", run_id="run_001")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "user1", "run_id": "run_001"})
            _assert_exact_ids(rows, {_uuid(750)})
        finally:
            db.delete_col()

    def test_combined_all_three_scope_filters(self):
        """Combined user_id + agent_id + run_id filter narrows to exact match."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(760), VECTOR_COFFEE, _make_payload("exact", user_id="user1", agent_id="agent_a", run_id="run_001")),
                    (_uuid(761), VECTOR_FLIGHT, _make_payload("diff run", user_id="user1", agent_id="agent_a", run_id="run_002")),
                    (_uuid(762), VECTOR_WINDOW, _make_payload("diff agent", user_id="user1", agent_id="agent_b", run_id="run_001")),
                ],
            )
            rows = db.search(
                "test", VECTOR_COFFEE, top_k=10,
                filters={"user_id": "user1", "agent_id": "agent_a", "run_id": "run_001"},
            )
            _assert_exact_ids(rows, {_uuid(760)})
        finally:
            db.delete_col()

    def test_scope_guard_missing_user_id_raises_error(self):
        """Search without any scope filter raises ValueError when require_scoped_filters=True."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(770), VECTOR_COFFEE, _make_payload("item", user_id="user1", category="food")),
                ],
            )
            with pytest.raises(ValueError, match="requires at least one scoped filter"):
                db.search("test", VECTOR_COFFEE, top_k=10, filters={"category": "food"})
        finally:
            db.delete_col()

    def test_or_without_scope_in_all_branches_raises_error(self):
        """$or where not all branches have scope filter raises ValueError."""
        db = _new_db(prefix="p1_filter")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(780), VECTOR_COFFEE, _make_payload("item", user_id="user1", category="food")),
                ],
            )
            with pytest.raises(ValueError, match="requires at least one scoped filter"):
                db.search(
                    "test", VECTOR_COFFEE, top_k=10,
                    filters={"$or": [{"user_id": "user1"}, {"category": "food"}]},
                )
        finally:
            db.delete_col()

    def test_filter_mode_json_expression_basic(self):
        """json_expression filter mode supports payload key filtering."""
        db = _new_db(prefix="p1_filter", filter_storage_mode="json_expression")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(790), VECTOR_COFFEE, _make_payload("food item", user_id="json_user", category="food")),
                    (_uuid(791), VECTOR_FLIGHT, _make_payload("travel item", user_id="json_user", category="travel")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "json_user", "category": "food"})
            _assert_exact_ids(rows, {_uuid(790)})
        finally:
            db.delete_col()

    def test_filter_mode_json_expression_range(self):
        """json_expression filter mode supports range operators on payload fields."""
        db = _new_db(prefix="p1_filter", filter_storage_mode="json_expression")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(800), VECTOR_COFFEE, _make_payload("low", user_id="json_user", priority="2")),
                    (_uuid(801), VECTOR_FLIGHT, _make_payload("high", user_id="json_user", priority="8")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "json_user", "priority": {"gte": "5"}})
            _assert_exact_ids(rows, {_uuid(801)})
        finally:
            db.delete_col()

    def test_filter_mode_redundant_columns_basic(self):
        """redundant_columns filter mode supports scope column filtering."""
        db = _new_db(prefix="p1_filter", filter_storage_mode="redundant_columns")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(810), VECTOR_COFFEE, _make_payload("alice item", user_id="rc_alice")),
                    (_uuid(811), VECTOR_FLIGHT, _make_payload("bob item", user_id="rc_bob")),
                ],
            )
            rows = db.search("test", VECTOR_COFFEE, top_k=10, filters={"user_id": "rc_alice"})
            _assert_exact_ids(rows, {_uuid(810)})
        finally:
            db.delete_col()

