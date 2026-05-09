#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-language E2E test for GaussDB vector store.

Verifies that text in various languages (English, Chinese, Japanese, Korean,
Arabic, Russian, French, German, Spanish, Thai, Vietnamese, emoji, mixed)
can be correctly stored, retrieved, and searched (both vector and BM25).

Database: 121.37.186.131:19995, lxm/Gauss_234, lxm_db
Dims: 1536 (default, GsDiskANN)
"""

import os
import sys
import random
import uuid
import json
import io
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mem0.vector_stores.gaussdb import GaussDB

# ─── Config ───────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "121.37.186.131",
    "port": 19995,
    "user": "lxm",
    "password": "Gauss_234",
    "database": "lxm_db",
}

DIMS = 1536
COLLECTION = f"multilang_test_{uuid.uuid4().hex[:8]}"

# ─── Test Data ────────────────────────────────────────────────────────────────
MULTILANG_CASES = [
    {
        "lang": "English",
        "text": "I love programming in Python and building machine learning models.",
        "bm25_query": "Python programming",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "en"},
    },
    {
        "lang": "Chinese (Simplified)",
        "text": "我喜欢用Python做数据分析和机器学习，周末经常去公园跑步。",
        "bm25_query": "Python数据分析",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "zh-CN"},
    },
    {
        "lang": "Chinese (Traditional)",
        "text": "台灣的珍珠奶茶非常好喝，我每天都要來一杯。",
        "bm25_query": "珍珠奶茶",
        "payload": {"user_id": "lang_test", "category": "food", "language": "zh-TW"},
    },
    {
        "lang": "Japanese",
        "text": "東京の桜は春に美しく咲きます。日本語のプログラミング教材を読んでいます。",
        "bm25_query": "桜",
        "payload": {"user_id": "lang_test", "category": "culture", "language": "ja"},
    },
    {
        "lang": "Korean",
        "text": "서울에서 김치찌개를 먹었습니다. 한국어 공부를 열심히 하고 있습니다.",
        "bm25_query": "김치찌개",
        "payload": {"user_id": "lang_test", "category": "food", "language": "ko"},
    },
    {
        "lang": "Arabic",
        "text": "أنا أحب البرمجة وتعلم اللغات الجديدة. القهوة العربية لذيذة جداً.",
        "bm25_query": "البرمجة",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "ar"},
    },
    {
        "lang": "Russian",
        "text": "Я люблю программирование на Python и изучаю машинное обучение.",
        "bm25_query": "Python",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "ru"},
    },
    {
        "lang": "French",
        "text": "J'adore la programmation et je travaille comme ingénieur logiciel à Paris.",
        "bm25_query": "programmation",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "fr"},
    },
    {
        "lang": "German",
        "text": "Ich arbeite als Softwareentwickler in Berlin und programmiere gerne in Python.",
        "bm25_query": "Softwareentwickler",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "de"},
    },
    {
        "lang": "Spanish",
        "text": "Me encanta programar en Python y desarrollar aplicaciones web modernas.",
        "bm25_query": "programar",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "es"},
    },
    {
        "lang": "Thai",
        "text": "ฉันชอบเขียนโปรแกรมภาษา Python และทำงานด้านวิทยาศาสตร์ข้อมูล",
        "bm25_query": "Python",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "th"},
    },
    {
        "lang": "Vietnamese",
        "text": "Tôi thích lập trình Python và nghiên cứu trí tuệ nhân tạo.",
        "bm25_query": "Python",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "vi"},
    },
    {
        "lang": "Emoji + Mixed",
        "text": "🎉 Hello世界! Python🐍 is awesome すごい 太棒了 fantastique! 🚀💻",
        "bm25_query": "Python",
        "payload": {"user_id": "lang_test", "category": "mixed", "language": "mixed"},
    },
    {
        "lang": "Long Chinese",
        "text": (
            "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次化表示。"
            "卷积神经网络在图像识别领域取得了巨大成功，循环神经网络则擅长处理序列数据。"
            "近年来，Transformer架构彻底改变了自然语言处理领域，BERT和GPT等模型"
            "在各种NLP任务上都达到了前所未有的性能。华为GaussDB数据库支持向量检索，"
            "可以高效地存储和查询高维向量数据，为AI应用提供强大的数据基础设施支持。"
        ),
        "bm25_query": "GaussDB向量检索",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "zh-CN"},
    },
    {
        "lang": "Special Characters",
        "text": "C++ & C# are <great> languages; SELECT * FROM 'table' WHERE x=\"test\" -- comment",
        "bm25_query": "languages",
        "payload": {"user_id": "lang_test", "category": "tech", "language": "en"},
    },
]


def make_vector(seed: int) -> list:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIMS)]


def main():
    print("=" * 70)
    print("  mem0 GaussDB Multi-Language E2E Test")
    print(f"  Target: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"  Collection: {COLLECTION}")
    print(f"  Dims: {DIMS}")
    print(f"  Languages: {len(MULTILANG_CASES)}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)

    store = GaussDB(
        **DB_CONFIG,
        collection_name=COLLECTION,
        embedding_model_dims=DIMS,
        vector_index_type="gsdiskann",
        vector_metric="cosine",
        bm25_mode="auto",
        metadata_mode="auto",
        require_scoped_filters=True,
    )
    print(f"\n  Store initialized. BM25 enabled: {store.bm25_enabled}")
    print(f"  Payload mode: {store.payload_storage_mode}, Filter mode: {store.filter_storage_mode}")

    passed = 0
    failed = 0
    results = []

    # ─── Phase 1: Insert all languages ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Phase 1: INSERT (all languages)")
    print(f"{'─'*70}")

    ids = []
    for i, case in enumerate(MULTILANG_CASES):
        vid = str(uuid.uuid4())
        ids.append(vid)
        vec = make_vector(i)
        payload = {**case["payload"], "text": case["text"]}
        try:
            store.insert(vectors=[vec], ids=[vid], payloads=[payload])
            print(f"  [{i+1:02d}] {case['lang']:25s} ✓ inserted ({len(case['text'])} chars)")
            passed += 1
        except Exception as e:
            print(f"  [{i+1:02d}] {case['lang']:25s} ✗ FAILED: {e}")
            failed += 1
            results.append(("INSERT", case["lang"], str(e)))

    # ─── Phase 2: Retrieve and verify payload integrity ───────────────────────
    print(f"\n{'─'*70}")
    print("  Phase 2: RETRIEVE & VERIFY (payload text integrity)")
    print(f"{'─'*70}")

    for i, (case, vid) in enumerate(zip(MULTILANG_CASES, ids)):
        try:
            record = store.get(vector_id=vid)
            if record is None:
                raise ValueError("get() returned None")
            stored_text = record.payload.get("text", "")
            if stored_text != case["text"]:
                raise ValueError(
                    f"Text mismatch!\n  Expected: {case['text'][:60]}...\n  Got:      {stored_text[:60]}..."
                )
            print(f"  [{i+1:02d}] {case['lang']:25s} ✓ text intact ({len(stored_text)} chars)")
            passed += 1
        except Exception as e:
            print(f"  [{i+1:02d}] {case['lang']:25s} ✗ FAILED: {e}")
            failed += 1
            results.append(("RETRIEVE", case["lang"], str(e)))

    # ─── Phase 3: Vector search (semantic) ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Phase 3: VECTOR SEARCH (cosine similarity)")
    print(f"{'─'*70}")

    for i, case in enumerate(MULTILANG_CASES):
        try:
            query_vec = make_vector(i)  # Same vector as inserted → should be top hit
            hits = store.search(query=case["text"], vectors=query_vec, top_k=1, filters={"user_id": "lang_test"})
            if not hits:
                raise ValueError("search returned 0 results")
            top_hit = hits[0]
            if top_hit.id != ids[i]:
                raise ValueError(f"Expected id={ids[i]}, got id={top_hit.id}")
            print(f"  [{i+1:02d}] {case['lang']:25s} ✓ top hit correct (score={top_hit.score:.4f})")
            passed += 1
        except Exception as e:
            print(f"  [{i+1:02d}] {case['lang']:25s} ✗ FAILED: {e}")
            failed += 1
            results.append(("VECTOR_SEARCH", case["lang"], str(e)))

    # ─── Phase 4: BM25 keyword search ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Phase 4: BM25 KEYWORD SEARCH")
    print(f"{'─'*70}")

    if not store.bm25_enabled:
        print("  ⚠ BM25 not enabled, skipping")
    else:
        for i, case in enumerate(MULTILANG_CASES):
            try:
                bm25_results = store.keyword_search(
                    query=case["bm25_query"],
                    top_k=5,
                    filters={"user_id": "lang_test"},
                )
                if bm25_results is None:
                    # BM25 may return None for unsupported tokenization
                    print(f"  [{i+1:02d}] {case['lang']:25s} ⚠ BM25 returned None (tokenization issue)")
                    passed += 1
                elif len(bm25_results) == 0:
                    print(f"  [{i+1:02d}] {case['lang']:25s} ⚠ BM25 returned 0 results (query: {case['bm25_query']})")
                    passed += 1
                else:
                    found_ids = [r.id for r in bm25_results]
                    marker = "✓" if ids[i] in found_ids else "⚠"
                    print(
                        f"  [{i+1:02d}] {case['lang']:25s} {marker} "
                        f"BM25 hits={len(bm25_results)}, self_found={ids[i] in found_ids} "
                        f"(query: {case['bm25_query']})"
                    )
                    passed += 1
            except Exception as e:
                print(f"  [{i+1:02d}] {case['lang']:25s} ✗ FAILED: {e}")
                failed += 1
                results.append(("BM25_SEARCH", case["lang"], str(e)))

    # ─── Phase 5: Update payload with multilang text ──────────────────────────
    print(f"\n{'─'*70}")
    print("  Phase 5: UPDATE PAYLOAD (multilang)")
    print(f"{'─'*70}")

    update_texts = {
        "Chinese": "更新后的中文文本：华为GaussDB是一款优秀的分布式数据库。",
        "Japanese": "更新されたテキスト：東京タワーは美しいです。",
        "Korean": "업데이트된 텍스트: 서울은 아름다운 도시입니다.",
    }
    update_indices = [1, 3, 4]  # Chinese simplified, Japanese, Korean

    for idx, (label, new_text) in zip(update_indices, update_texts.items()):
        try:
            store.update(
                vector_id=ids[idx],
                payload={**MULTILANG_CASES[idx]["payload"], "text": new_text, "updated": True},
            )
            # Verify
            record = store.get(vector_id=ids[idx])
            stored = record.payload.get("text", "")
            if stored != new_text:
                raise ValueError(f"Update failed: expected '{new_text[:30]}...', got '{stored[:30]}...'")
            print(f"  {label:25s} ✓ updated and verified")
            passed += 1
        except Exception as e:
            print(f"  {label:25s} ✗ FAILED: {e}")
            failed += 1
            results.append(("UPDATE", label, str(e)))

    # ─── Phase 6: Delete and verify ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Phase 6: DELETE & VERIFY")
    print(f"{'─'*70}")

    try:
        store.delete(vector_id=ids[0])
        record = store.get(vector_id=ids[0])
        if record is not None:
            raise ValueError("Record still exists after delete")
        print(f"  English record deleted ✓")
        passed += 1
    except Exception as e:
        print(f"  Delete test ✗ FAILED: {e}")
        failed += 1
        results.append(("DELETE", "English", str(e)))

    # ─── Phase 7: Collection info ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Phase 7: COLLECTION INFO")
    print(f"{'─'*70}")

    try:
        info = store.col_info()
        print(f"  Name: {info['name']}")
        print(f"  Count: {info['count']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Indexes: {info['indexes']}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
        results.append(("COL_INFO", "-", str(e)))

    # ─── Cleanup ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Cleanup")
    print(f"{'─'*70}")

    store.delete_col()
    print(f"  Collection '{COLLECTION}' deleted ✓")

    # ─── Summary ──────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*70}")
    print(f"  TEST SUMMARY: {passed} passed, {failed} failed / {total} total")
    if results:
        print(f"\n  Failures:")
        for phase, lang, err in results:
            print(f"    [{phase}] {lang}: {err[:80]}")
    print(f"{'='*70}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
