#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import asyncio
import queue
import threading
import time


class TokenBucket:
    """Thread-safe rate limiter with dynamic token adjustments."""

    def __init__(self, rate: float, capacity: int) -> None:
        self.capacity = capacity  # Max tokens
        self.rate = rate  # Tokens per second
        self.tokens: float = capacity  # Start full
        self.lock = threading.Lock()
        self.last_refill = time.time()
        self.condition_lock = threading.Condition(self.lock)
        self.async_lock = asyncio.Lock()
        self.waiting_threads: queue.Queue[int] = queue.Queue()

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        if new_tokens >= 1:  # Only update if at least one token is added
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    def acquire(self) -> None:
        """Wait for a token and process threads in correct order."""
        thread_id = threading.get_ident()

        with self.condition_lock:
            # Add to queue if not already in front
            if (
                self.waiting_threads.empty()
                or self.waiting_threads.queue[-1] != thread_id
            ):
                self.waiting_threads.put(thread_id)

            while True:
                self.refill()

                # Allow thread to proceed only if it's at the
                # front of the queue and tokens are available
                if self.tokens >= 1 and self.waiting_threads.queue[0] == thread_id:
                    self.waiting_threads.get()  # Remove from queue
                    self.tokens -= 1  # Consume token
                    self.condition_lock.notify()  # Wake next in line
                    return

                # Wait only until the next expected refill time
                next_refill = self.last_refill + (1 / self.rate)
                wait_time_float = max(0.0, next_refill - time.time())
                self.condition_lock.wait(wait_time_float)

    async def acquire_async(self) -> None:
        """Asynchronous acquire: Wait until a token is available."""
        async with self.async_lock:
            while self.tokens < 1:
                self.refill()
                wait_time = (1 / self.rate) if self.tokens < 1 else 0
                await asyncio.sleep(wait_time)
            self.tokens -= 1

    def adjust_tokens(self, remaining_tokens: int) -> None:
        """Adjust token count based on RateLimit-Remaining."""
        with self.lock:
            self.tokens = min(self.capacity, remaining_tokens)

    async def async_adjust_tokens(self, remaining_tokens: int) -> None:
        """Adjust token count based on RateLimit-Remaining."""
        async with self.async_lock:
            self.tokens = min(self.capacity, remaining_tokens)
