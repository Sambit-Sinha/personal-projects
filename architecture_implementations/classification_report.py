"""
Implement classification_report for binary classification.

Return:
{
  "confusion_matrix": {
      "tp": int,
      "fp": int,
      "tn": int,
      "fn": int
  },
  "precision": float,
  "recall": float,
  "f1_score": float
}

Rules:
- Only binary labels (0,1)
- Raise ValueError if lengths mismatch
- If both lists empty -> return all 0
- If denominator is 0 -> metric = 0.0
- Round all float metrics to 4 decimals
"""

from typing import List, Dict


def classification_report(y_true: List[int], y_pred: List[int]) -> Dict:
    # TODO
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch!")
    if len(y_true) == 0 and len(y_pred) == 0:
        return {
  "confusion_matrix": {
      "tp": 0,
      "fp": 0,
      "tn": 0,
      "fn": 0
  },
  "precision": 0.0,
  "recall": 0.0,
  "f1_score": 0.0
}
    dc = {
  "confusion_matrix": {
      "tp": 0,
      "fp": 0,
      "tn": 0,
      "fn": 0
  },
  "precision": float,
  "recall": float,
  "f1_score": float
}
    # dc = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'f1':0}
    for idx, ele in enumerate(y_true):
        if ele == 1:
            # print(y_pred[idx] y_true)
            dc["confusion_matrix"]['tp'] += int(y_pred[idx] == ele)
            dc["confusion_matrix"]['fn'] += int(y_pred[idx] != ele)
        if ele == 0:
            dc["confusion_matrix"]['tn'] += int(y_pred[idx] == ele)
            dc["confusion_matrix"]['fp'] += int(y_pred[idx] != ele)
    dc['precision'] = (dc["confusion_matrix"]['tp'])/(dc["confusion_matrix"]['tp'] + dc["confusion_matrix"]['fp'])
    dc['recall'] = (dc["confusion_matrix"]['tp'])/(dc["confusion_matrix"]['tp'] + dc["confusion_matrix"]['fn']) 
    if dc['precision'] !=0 and dc['recall'] != 0:
        dc['f1_score'] = 2*(dc['precision']*dc['recall'])/(dc['precision']+dc['recall'])
    else:
        dc['f1_score'] = 0.0
    
    # print(dc)
    return dc


def run_test(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name} -> {type(e).__name__}: {e}")
        return False


def test_classification_report():
    print("\nChallenge 1: Classification Metrics")
    print("-" * 40)
    results = []

    def tc1():
        result = classification_report([1, 0, 1, 1], [1, 0, 0, 1])
        assert result["confusion_matrix"]["tp"] == 2
        assert result["confusion_matrix"]["fp"] == 0
        assert result["confusion_matrix"]["fn"] == 1
        assert abs(result["precision"] - 1.0) < 0.0001
        assert abs(result["recall"] - 0.6667) < 0.0001

    def tc2():
        result2 = classification_report([], [])
        assert result2["precision"] == 0.0

    def tc3():
        try:
            classification_report([1, 0], [1])
            assert False, "Expected ValueError"
        except ValueError:
            pass

    results.append(run_test("Confusion matrix, precision & recall", tc1))
    results.append(run_test("Empty input returns all zeros", tc2))
    results.append(run_test("Length mismatch raises ValueError", tc3))

    print("-" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} test cases passed\n")


if __name__ == "__main__":
    test_classification_report()
