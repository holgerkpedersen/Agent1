"""Message bus implementation for inter-agent communication.

This module provides a centralized message routing system that allows agents
to communicate asynchronously through channels and topics. The MessageBus
supports pub/sub patterns, direct messaging, broadcast notifications, and
request/response interactions with optional timeout handling.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Enumeration of supported message types."""

    PUBLISH = "publish"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DIRECT_MESSAGE = "direct_message"
    BROADCAST = "broadcast"
    REQUEST = "request"
    RESPONSE = "response"


@dataclass
class Message:
    """Represents a single message exchanged between agents.

    Attributes:
        id: Unique identifier for the message.
        type: The category of message being sent.
        source: Identifier of the sending agent or component.
        target: Optional recipient identifier (None for broadcasts/topics).
        topic: Topic name used for pub/sub routing; None if not applicable.
        channel: Channel name used for direct messaging; None if not applicable.
        payload: Arbitrary data carried by the message.
        timestamp: UTC timestamp when the message was created.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.DIRECT_MESSAGE
    source: str = ""
    target: Optional[str] = None
    topic: Optional[str] = None
    channel: Optional[str] = None
    payload: Any = None
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class Subscription:
    """Represents an active subscription to a topic or channel.

    Attributes:
        subscriber_id: Identifier of the subscribing agent.
        callback: Asynchronous callable invoked when matching messages arrive.
        topics: Set of topics this subscription listens on (empty for all).
        channels: Set of direct message channels this subscription listens on.
    """

    subscriber_id: str = ""
    callback: Callable[[Message], Coroutine[Any, Any, None]] = lambda msg: asyncio.sleep(0)  # noqa: E731
    topics: Set[str] = field(default_factory=set)
    channels: Set[str] = field(default_factory=set)


class MessageBus:
    """Centralized asynchronous message routing system.

    The bus maintains subscriptions keyed by topic and channel, routes incoming
    messages to matching subscribers, and provides convenience methods for common
    communication patterns such as publish/subscribe, direct messaging, broadcast,
    request/response, and subscription lifecycle management.

    Thread-safety note: This implementation is designed for use within a single
    asyncio event loop. It does not provide cross-thread synchronization; callers
    should ensure all interactions occur from the same thread or properly marshal
    calls into the owning event loop.
    """

    def __init__(self, name: str = "default") -> None:
        self.name: str = name
        # topic_name -> list of subscriptions listening on that topic (or wildcard)
        self._topic_subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        # channel_name -> list of subscriptions listening on that channel
        self._channel_subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        # subscriber_id -> subscription metadata for quick lookup/unsubscribe
        self._subscriber_index: Dict[str, Subscription] = {}
        # pending request id -> (future, expiry_task) for timeout handling
        self._pending_requests: Dict[str, Tuple[asyncio.Future, asyncio.Task]] = {}

    async def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
        topics: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
    ) -> Subscription:
        """Register a new subscription with the message bus.

        Args:
            subscriber_id: Unique identifier for the subscribing agent.
            callback: Async function called when relevant messages arrive.
            topics: List of topic names to listen on; empty/None means wildcard (all).
            channels: List of channel names to listen on; empty/None means none.

        Returns:
            The created Subscription object, also stored internally for lookup.

        Raises:
            ValueError: If subscriber_id already exists with a conflicting subscription.
        """
        if not callable(callback):
            raise TypeError("callback must be an async callable")

        topic_set = set(topics) if topics else set()
        channel_set = set(channels) if channels else set()

        subscription = Subscription(
            subscriber_id=subscriber_id,
            callback=callback,
            topics=topic_set,
            channels=channel_set,
        )

        # Register under wildcard topic ("*") so all messages go there too.
        self._topic_subscriptions["*"].append(subscription)
        for topic in topic_set:
            self._topic_subscriptions[topic].append(subscription)
        for channel in channel_set:
            self._channel_subscriptions[channel].append(subscription)

        if subscriber_id in self._subscriber_index and self._subscriber_index[subscriber_id] is not subscription:
            logger.warning(
                "Replacing existing subscription for subscriber '%s'", subscriber_id
            )

        self._subscriber_index[subscriber_id] = subscription
        logger.debug("Subscriber '%s' registered (topics=%s, channels=%s)", subscriber_id, topic_set, channel_set)
        return subscription

    async def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscription identified by its subscriber ID.

        Args:
            subscriber_id: Identifier of the subscribing agent to remove.

        Returns:
            True if a subscription was removed; False otherwise.
        """
        subscription = self._subscriber_index.pop(subscriber_id, None)
        if subscription is None:
            logger.debug("Unsubscribe requested for '%s' but no active subscription found", subscriber_id)
            return False

        # Remove from topic lists including wildcard bucket.
        all_topic_buckets = ["*"] + list(subscription.topics)
        for bucket in all_topic_buckets:
            subs_list = self._topic_subscriptions.get(bucket, [])
            try:
                subs_list.remove(subscription)
            except ValueError:
                pass  # Already removed or not present.

        for channel in subscription.channels:
            subs_list = self._channel_subscriptions.get(channel, [])
            try:
                subs_list.remove(subscription)
            except ValueError:
                pass

        logger.debug("Subscriber '%s' unsubscribed", subscriber_id)
        return True

    async def publish(
        self, topic: str, payload: Any = None, source: str = "", exclude: Optional[Set[str]] = None
    ) -> Message:
        """Publish a message to all subscribers of the given topic.

        Wildcard ("*") subscriptions receive every published message regardless
        of topic. Subscribers explicitly listed in `exclude` are skipped.

        Args:
            topic: Name of the topic to publish on.
            payload: Data to include in the message body.
            source: Identifier of the publishing agent/component.
            exclude: Optional set of subscriber IDs to skip delivery for.

        Returns:
            The dispatched Message object.
        """
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")

        message = Message(
            type=MessageType.PUBLISH,
            source=source,
            target=None,
            topic=topic,
            payload=payload,
        )

        exclude_set = exclude or set()

        # Deliver to wildcard listeners first (they receive everything).
        recipients: List[Subscription] = []
        seen_ids: Set[str] = set()

        for bucket_name in ("*", topic):
            for sub in self._topic_subscriptions.get(bucket_name, []):
                if sub.subscriber_id not in exclude_set and sub.subscriber_id not in seen_ids:
                    recipients.append(sub)
                    seen_ids.add(sub.subscriber_id)

        await self._dispatch_to(recipients, message)
        logger.debug("Published topic '%s' from '%s' to %d recipient(s)", topic, source, len(recipients))
        return message

    async def send_direct_message(
        self, channel: str, target: str, payload: Any = None, source: str = ""
    ) -> Message:
        """Send a direct/private message over a named channel to one or more recipients.

        All subscriptions listening on the specified channel will receive the
        message; callers typically use unique channel names per conversation pair
        for true private messaging semantics.

        Args:
            channel: Name of the communication channel.
            target: Identifier of intended recipient(s) (stored as metadata only).
            payload: Message body content.
            source: Sender identifier.

        Returns:
            The dispatched Message object.
        """
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("channel must be a non-empty string")

        message = Message(
            type=MessageType.DIRECT_MESSAGE,
            source=source,
            target=target,
            channel=channel,
            payload=payload,
        )

        recipients: List[Subscription] = []
        seen_ids: Set[str] = set()
        for sub in self._channel_subscriptions.get(channel, []):
            if sub.subscriber_id not in seen_ids:
                recipients.append(sub)
                seen_ids.add(sub.subscriber_id)

        await self._dispatch_to(recipients, message)
        logger.debug("Direct message on channel '%s' from '%s' to %d recipient(s)", channel, source, len(recipients))
        return message

    async def broadcast(self, payload: Any = None, source: str = "", exclude: Optional[Set[str]] = None) -> Message:
        """Broadcast a notification to every active subscriber on the bus.

        Args:
            payload: Notification content.
            source: Originator identifier.
            exclude: Optional set of subscriber IDs to skip.

        Returns:
            The dispatched broadcast Message object.
        """
        message = Message(
            type=MessageType.BROADCAST,
            source=source,
            target=None,
            topic=None,
            channel=None,
            payload=payload,
        )

        exclude_set = exclude or set()
        recipients: List[Subscription] = []
        seen_ids: Set[str] = set()
        for sub in self._topic_subscriptions.get("*", []):
            if sub.subscriber_id not in exclude_set and sub.subscriber_id not in seen_ids:
                recipients.append(sub)
                seen_ids.add(sub.subscriber_id)

        await self._dispatch_to(recipients, message)
        logger.debug("Broadcast from '%s' to %d recipient(s)", source, len(recipients))
        return message

    async def request(
        self, target: str, payload: Any = None, source: str = "", timeout: float = 30.0
    ) -> Message:
        """Send a request and await its corresponding response via the bus.

        This method publishes a REQUEST-typed message to wildcard subscribers,
        registers an internal future keyed by message ID so that any agent can
        respond using `respond()`, and waits up to `timeout` seconds for arrival.

        Args:
            target: Intended recipient identifier (metadata).
            payload: Request parameters/data.
            source: Requesting agent identifier.
            timeout: Maximum wait time in seconds before raising TimeoutError.

        Returns:
            The Response Message returned by the responder.

        Raises:
            asyncio.TimeoutError: If no response arrives within `timeout`.
        """
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")

        message = Message(
            type=MessageType.REQUEST,
            source=source,
            target=target,
            topic=None,
            channel=None,
            payload=payload,
        )

        loop = asyncio.get_event_loop()
        future: asyncio.Future[Message] = loop.create_future()
        expiry_task = loop.create_task(asyncio.sleep(timeout))
        self._pending_requests[message.id] = (future, expiry_task)

        await self.publish(topic="*", payload=message.payload, source=source)  # noqa: SIM106 - publish raises on bad topic but "*" is valid.

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Request '%s' timed out after %.2fs waiting for response", message.id, timeout)
            raise
        finally:
            self._pending_requests.pop(message.id, None)

        return response

    async def respond(self, request_id: str, payload: Any = None, source: str = "") -> Message:
        """Deliver a response to the originator of a previously issued request.

        Args:
            request_id: Identifier matching an outstanding REQUEST message ID.
            payload: Response data/result.
            source: Responding agent identifier.

        Returns:
            The dispatched RESPONSE Message object, or None if no pending request exists.
        """
        entry = self._pending_requests.get(request_id)
        if entry is None:
            logger.warning("No pending request found for id '%s'; response not routed", request_id)
            return None  # type: ignore[return-value]

        future, expiry_task = entry
        message = Message(
            type=MessageType.RESPONSE,
            source=source,
            target=None,
            topic=request_id,
            channel=None,
            payload=payload,
        )

        if not future.done():
            future.set_result(message)

        expiry_task.cancel()
        await self.publish(topic="*", payload=message.payload, source=source)  # noqa: SIM106.

        logger.debug("Response sent for request '%s' from '%s'", request_id, source)
        return message

    async def _dispatch_to(self, recipients: List[Subscription], message: Message) -> None:
        """Route a single message concurrently to all matching subscriptions."""
        if not recipients:
            logger.debug("No subscribers matched for message '%s'", message.id)
            return

        tasks = []
        for sub in recipients:
            try:
                task = asyncio.ensure_future(sub.callback(message))
                tasks.append(task)
            except Exception as exc:  # noqa: BLE001 - callback invocation failures shouldn't halt routing.
                logger.error("Callback error for subscriber '%s': %s", sub.subscriber_id, exc)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):  # noqa: BLE001.
                    logger.error("Subscriber callback raised exception during dispatch: %s", result)

    def get_subscriber_count(self) -> int:
        """Return the total number of distinct active subscribers."""
        return len(self._subscriber_index)

    def list_topics(self) -> List[str]:
        """List all topic names currently registered on the bus (excluding wildcard)."""
        keys = [k for k in self._topic_subscriptions.keys() if k != "*"]
        return sorted(keys)

    def list_channels(self) -> List[str]:
        """List all channel names currently registered on the bus."""
        return sorted(self._channel_subscriptions.keys())

    async def shutdown(self) -> None:
        """Cancel any pending request expiry timers and clear internal state."""
        logger.info("Shutting down message bus '%s'", self.name)
        for future, task in self._pending_requests.values():
            if not task.done():
                task.cancel()
            if not future.done():
                future.cancel()

        self._pending_requests.clear()
        self._subscriber_index.clear()
        self._topic_subscriptions.clear()
        self._channel_subscriptions.clear()


__all__: Tuple[str, ...] = (
    "MessageBus",
    "Message",
    "Subscription",
    "MessageType",
)