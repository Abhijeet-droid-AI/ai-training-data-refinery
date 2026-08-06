from src.preprocessing.pipeline import PreprocessingPipeline


def test_pipeline():
    documents = [
        {
            "id": 1,
            "title": "Python",
            "text": "<div>Python&nbsp;&nbsp;is <b>awesome</b></div>"
        }
    ]

    pipeline = PreprocessingPipeline()

    processed = pipeline.process(documents)

    assert len(processed) == 1

    assert processed[0]["text"] == "Python is awesome"

    assert processed[0]["metadata"]["html_removed"] is True

    assert processed[0]["metadata"]["language"] == "en"