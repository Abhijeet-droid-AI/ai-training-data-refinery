from src.profiling.profiler import DatasetProfiler

def test_profile():
    docs = [
        {
            "id": 1,
            "title": "AI",
            "text": "Artificial Intelligence"
        }
    ]

    report = DatasetProfiler(docs).profile()

    assert report["total_documents"] == 1