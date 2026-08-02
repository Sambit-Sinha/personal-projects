"""
Implement rank_documents for hybrid retrieval ranking.

final_score = alpha * embedding_score + (1 - alpha) * keyword_score

Rules:
- Sort descending by final_score
- Return top k documents
- Stable sort (preserve order if tie)
- If k > number of docs -> return all sorted
- Raise ValueError if:
    - input lengths mismatch
    - alpha not in [0,1]
"""

from typing import List


def rank_documents(
    docs: List[str],
    embedding_scores: List[float],
    keyword_scores: List[float],
    alpha: float,
    k: int
) -> List[str]:
    # TODO

    if len(embedding_scores) != len(keyword_scores) or len(embedding_scores)!=len(docs) or len(keyword_scores)!=len(docs) or not(0.0 <= alpha <= 1.0):
        raise ValueError
    
    ldek = []
    for i, doc in enumerate(docs):
        final_score = alpha*embedding_scores[i] + (1-alpha)*keyword_scores[i]
        ldek.append([doc,final_score])

    ldek = sorted(ldek,key=lambda x: x[1],reverse=True)
    ldek_topk = [i[0] for i in ldek[:k]]
    return ldek_topk


def run_test(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name} -> {type(e).__name__}: {e}")
        return False


def test_rank_documents():
    print("\nChallenge 3: Hybrid Document Ranking")
    print("-" * 40)
    results = []

    docs = ["docA", "docB", "docC"]
    embedding = [0.9, 0.6, 0.75]
    keyword = [0.2, 0.9, 0.5]

    def tc1():
        result = rank_documents(docs, embedding, keyword, 0.7, 2)
        assert len(result) == 2
        assert result[0] in ["docA", "docB"]

    def tc2():
        try:
            rank_documents(docs, embedding, keyword, 1.5, 2)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    results.append(run_test("Top-k ranking with hybrid scores", tc1))
    results.append(run_test("Invalid alpha raises ValueError", tc2))

    print("-" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} test cases passed\n")


if __name__ == "__main__":
    test_rank_documents()