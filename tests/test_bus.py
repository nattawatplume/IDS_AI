import asyncio

import pytest

from idsai.bus import TOPICS, Bus


def test_topics_are_the_four_of_the_brief():
    assert set(TOPICS) == {"bars", "scores", "alerts", "health"}


def test_each_subscriber_gets_its_own_queue_of_each_message():
    bus = Bus()
    a, b = bus.subscribe("alerts"), bus.subscribe("alerts")
    other = bus.subscribe("bars")
    bus.publish("alerts", 1)
    assert a.get_nowait() == 1 and b.get_nowait() == 1
    assert other.empty()


def test_drop_oldest_counts_every_drop_per_subscriber():
    bus = Bus(maxsize=2)
    a, b = bus.subscribe("bars"), bus.subscribe("bars")
    for i in range(5):
        bus.publish("bars", i)
    assert [a.get_nowait(), a.get_nowait()] == [3, 4]
    assert [b.get_nowait(), b.get_nowait()] == [3, 4]
    assert bus.dropped["bars"] == 6
    assert bus.dropped["alerts"] == 0


def test_slow_subscriber_does_not_affect_a_drained_one():
    bus = Bus(maxsize=1)
    slow, fast = bus.subscribe("scores"), bus.subscribe("scores")
    for i in range(3):
        bus.publish("scores", i)
        assert fast.get_nowait() == i
    assert slow.get_nowait() == 2
    assert bus.dropped["scores"] == 2


def test_unknown_topic_raises():
    bus = Bus()
    with pytest.raises(ValueError):
        bus.subscribe("nope")
    with pytest.raises(ValueError):
        bus.publish("nope", 1)


def test_no_subscribers_is_not_an_error_and_not_a_drop():
    bus = Bus(maxsize=1)
    for i in range(3):
        bus.publish("health", i)
    assert bus.dropped["health"] == 0


def test_maxsize_must_be_positive():
    with pytest.raises(ValueError):
        Bus(maxsize=0)


def test_publish_wakes_a_waiting_consumer():
    async def main():
        bus = Bus()
        q = bus.subscribe("alerts")
        waiter = asyncio.create_task(q.get())
        await asyncio.sleep(0)
        bus.publish("alerts", "x")
        return await asyncio.wait_for(waiter, 1)

    assert asyncio.run(main()) == "x"


def test_unsubscribe_stops_delivery_and_drop_counting_and_is_idempotent():
    bus = Bus(maxsize=1)
    gone, kept = bus.subscribe("bars"), bus.subscribe("bars")
    bus.unsubscribe("bars", gone)
    bus.unsubscribe("bars", gone)
    bus.publish("bars", 1)
    assert gone.empty() and kept.get_nowait() == 1
    bus.publish("bars", 2)
    bus.publish("bars", 3)
    assert bus.dropped["bars"] == 1  # only the kept queue
    with pytest.raises(ValueError):
        bus.unsubscribe("nope", gone)
