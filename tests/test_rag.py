from support_rag import classify_intent_label, requires_live_data, _evidence


def test_intent_categories():
    cases = {"How do I request a refund?": "get_refund", "Where is my invoice?": "check_invoice",
             "I was charged twice": "payment_issue", "I forgot my password": "recover_password",
             "How do I track my order?": "track_order", "Can I speak to a human agent?": "contact_human_agent"}
    for question, expected in cases.items():
        assert classify_intent_label(question) == expected


def test_live_data_detection():
    assert requires_live_data("Where is my order right now?")
    assert requires_live_data("Has my package been delivered?")
    assert not requires_live_data("What is the return policy?")


def test_empty_evidence_fails_closed():
    level, _ = _evidence("What is the weather today?", [], [])
    assert level == "insufficient"
