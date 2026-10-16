from uuid import uuid4

from adaptive_dms.contracts import DriverEvent, EventSource, OwnerAdapter


def test_owner_adapter_is_bound_to_a_base_model() -> None:
    adapter = OwnerAdapter(
        owner_id=uuid4(),
        base_model_version="global-0.1.0",
        adapter_version="owner-1",
        attention_threshold=0.75,
    )
    assert adapter.min_event_duration_ms == 500


def test_driver_event_requires_a_known_event_type() -> None:
    event = DriverEvent(
        event_type="visual_distraction",
        confidence=0.95,
        source=EventSource.GLOBAL,
        timestamp_ms=500,
        model_version="global-0.1.0",
    )
    assert event.source is EventSource.GLOBAL
