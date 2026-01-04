from dataclasses import dataclass


@dataclass
class PIIResult:
    redacted_text: str
    redacted: bool
    entities: list


def redact_pii(text: str) -> PIIResult:
    """
    Uses Presidio if available; otherwise no-op.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results = analyzer.analyze(text=text, entities=None, language="en")
        if not results:
            return PIIResult(text, False, [])

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={"DEFAULT": {"type": "replace", "new_value": "<PII>"}},
        )

        ents = [{"type": r.entity_type, "start": r.start,
                 "end": r.end, "score": r.score} for r in results]
        return PIIResult(anonymized.text, True, ents)

    except Exception:
        # No presidio installed or runtime issue -> no-op
        return PIIResult(text, False, [])
