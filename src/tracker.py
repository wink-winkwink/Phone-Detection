"""
tracker.py - Pickup counting and time management logic.

Tracks when a person picks up a phone using bounding box intersection,
manages cooldowns, and enforces the hourly pickup limit.
"""

import time
from collections import deque


class PickupTracker:
    """Tracks phone pickups using bounding box proximity and time windows."""

    def __init__(self, hourly_limit=5, cooldown_seconds=4, time_window=3600):
        """
        Initialize the pickup tracker.

        Args:
            hourly_limit: Max pickups allowed within the time window.
            cooldown_seconds: Seconds to wait before counting another pickup.
            time_window: Time window in seconds (default: 3600 = 1 hour).
        """
        self.hourly_limit = hourly_limit
        self.cooldown_seconds = cooldown_seconds
        self.time_window = time_window

        self.is_holding = False
        self.pickup_timestamps = deque()
        self.last_pickup_time = 0

    def _boxes_overlap(self, box_a, box_b):
        """
        Check if two bounding boxes overlap or are near each other.

        Args:
            box_a: [x1, y1, x2, y2] bounding box.
            box_b: [x1, y1, x2, y2] bounding box.

        Returns:
            True if boxes overlap, False otherwise.
        """
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        return x1 < x2 and y1 < y2

    def _is_phone_near_person(self, persons, phones):
        """
        Check if any phone bounding box overlaps with any person bounding box.

        Args:
            persons: List of person bounding boxes.
            phones: List of phone bounding boxes.

        Returns:
            True if a phone is near/overlapping a person.
        """
        for phone in phones:
            for person in persons:
                if self._boxes_overlap(phone, person):
                    return True
        return False

    def _clean_old_timestamps(self):
        """Remove timestamps older than the time window."""
        current_time = time.time()
        while self.pickup_timestamps and (current_time - self.pickup_timestamps[0]) > self.time_window:
            self.pickup_timestamps.popleft()

    def update(self, detections):
        """
        Update the tracker state based on current frame detections.

        Args:
            detections: Dict with 'persons' and 'phones' bounding box lists.

        Returns:
            A dict with:
                - 'is_holding': bool, whether phone is currently held.
                - 'pickup_count': int, pickups in the current time window.
                - 'limit_exceeded': bool, whether the hourly limit is exceeded.
                - 'new_pickup': bool, whether a new pickup was just registered.
        """
        persons = detections.get('persons', [])
        phones = detections.get('phones', [])

        currently_holding = self._is_phone_near_person(persons, phones)
        new_pickup = False
        current_time = time.time()

        # Detect a new pickup: transition from not holding to holding
        if currently_holding and not self.is_holding:
            # Check cooldown
            if (current_time - self.last_pickup_time) >= self.cooldown_seconds:
                self.pickup_timestamps.append(current_time)
                self.last_pickup_time = current_time
                new_pickup = True

        self.is_holding = currently_holding

        # Clean old timestamps outside the time window
        self._clean_old_timestamps()

        pickup_count = len(self.pickup_timestamps)
        limit_exceeded = pickup_count > self.hourly_limit

        return {
            'is_holding': self.is_holding,
            'pickup_count': pickup_count,
            'limit_exceeded': limit_exceeded,
            'new_pickup': new_pickup,
        }

    def reset(self):
        """Reset the tracker state."""
        self.is_holding = False
        self.pickup_timestamps.clear()
        self.last_pickup_time = 0
