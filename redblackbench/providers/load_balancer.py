"""Load Balancer for multi-provider LLM distribution.

Provides:
- Round-robin and weighted load balancing across providers
- Message queue for managing high concurrency
- Automatic failover when providers fail
- Per-provider rate limiting and circuit breakers
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from collections import deque
import hashlib


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_PENDING = "least_pending"
    RANDOM = "random"
    FAILOVER = "failover"  # Primary with fallback


@dataclass
class ProviderStats:
    """Statistics for a single provider."""
    name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    pending_requests: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    is_healthy: bool = True
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return (self.total_latency / self.successful_requests) * 1000


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer.
    
    Attributes:
        strategy: Load balancing strategy
        max_queue_size: Maximum pending requests in queue
        queue_timeout: Timeout for queue waiting (seconds)
        max_concurrent_per_provider: Max concurrent requests per provider
        failure_threshold: Failures before marking provider unhealthy
        recovery_timeout: Seconds before retrying unhealthy provider
        weights: Provider weights for weighted strategy
    """
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    max_queue_size: int = 500
    queue_timeout: float = 120.0
    max_concurrent_per_provider: int = 5
    failure_threshold: int = 10
    recovery_timeout: float = 30.0
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueuedRequest:
    """A request waiting in the queue."""
    id: str
    system_prompt: str
    messages: List[dict]
    future: asyncio.Future
    enqueue_time: float
    priority: int = 0  # Higher = more priority


class MessageQueue:
    """Async message queue for managing request concurrency."""
    
    def __init__(self, max_size: int = 500, timeout: float = 120.0):
        self.max_size = max_size
        self.timeout = timeout
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._pending: Dict[str, QueuedRequest] = {}
        self._lock = asyncio.Lock()
        
        # Stats
        self.total_enqueued = 0
        self.total_processed = 0
        self.total_timeouts = 0
        self.total_dropped = 0
    
    async def enqueue(
        self,
        system_prompt: str,
        messages: List[dict],
        priority: int = 0,
    ) -> asyncio.Future:
        """Add a request to the queue.
        
        Args:
            system_prompt: System prompt
            messages: Message list
            priority: Request priority (higher = first)
            
        Returns:
            Future that will contain the response
        """
        request_id = hashlib.md5(
            f"{time.time()}{random.random()}".encode()
        ).hexdigest()[:12]
        
        future = asyncio.get_event_loop().create_future()
        request = QueuedRequest(
            id=request_id,
            system_prompt=system_prompt,
            messages=messages,
            future=future,
            enqueue_time=time.time(),
            priority=priority,
        )
        
        try:
            self._queue.put_nowait(request)
            async with self._lock:
                self._pending[request_id] = request
                self.total_enqueued += 1
        except asyncio.QueueFull:
            self.total_dropped += 1
            future.set_exception(RuntimeError("Queue full - request dropped"))
        
        return future
    
    async def dequeue(self) -> Optional[QueuedRequest]:
        """Get the next request from the queue.
        
        Returns:
            Next request or None if queue is empty
        """
        try:
            request = await asyncio.wait_for(
                self._queue.get(),
                timeout=1.0  # Quick poll
            )
            
            # Check if request has timed out
            if time.time() - request.enqueue_time > self.timeout:
                self.total_timeouts += 1
                request.future.set_exception(
                    asyncio.TimeoutError("Request timed out in queue")
                )
                async with self._lock:
                    self._pending.pop(request.id, None)
                return None
            
            return request
            
        except asyncio.TimeoutError:
            return None
    
    async def complete(self, request: QueuedRequest, result: str):
        """Mark a request as completed with result."""
        if not request.future.done():
            request.future.set_result(result)
        async with self._lock:
            self._pending.pop(request.id, None)
            self.total_processed += 1
    
    async def fail(self, request: QueuedRequest, error: Exception):
        """Mark a request as failed."""
        if not request.future.done():
            request.future.set_exception(error)
        async with self._lock:
            self._pending.pop(request.id, None)
    
    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "pending": self.pending_count,
            "total_enqueued": self.total_enqueued,
            "total_processed": self.total_processed,
            "total_timeouts": self.total_timeouts,
            "total_dropped": self.total_dropped,
        }


class LoadBalancedProvider:
    """Load balancer that distributes requests across multiple providers.
    
    Features:
    - Multiple load balancing strategies
    - Message queue for high concurrency
    - Automatic failover on provider failures
    - Per-provider statistics and health tracking
    """
    
    def __init__(
        self,
        providers: List[Any],  # List of provider instances
        config: Optional[LoadBalancerConfig] = None,
    ):
        """Initialize load balancer.
        
        Args:
            providers: List of LLM provider instances
            config: Load balancer configuration
        """
        self.config = config or LoadBalancerConfig()
        self._providers = providers
        self._provider_stats: Dict[str, ProviderStats] = {}
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        
        # Initialize stats for each provider
        for p in providers:
            name = getattr(p, 'provider_name', str(type(p).__name__))
            self._provider_stats[name] = ProviderStats(name=name)
        
        # Message queue (lazy init to avoid event loop issues)
        self._queue: Optional[MessageQueue] = None
        
        # Worker tasks
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Semaphores for per-provider concurrency (lazy init)
        # Track the event loop ID to detect when loop changes (e.g., from asyncio.run())
        self._provider_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._semaphore_loop_id: Optional[int] = None
    
    @property
    def provider_name(self) -> str:
        return "load_balanced"
    
    def _get_healthy_providers(self) -> List[Any]:
        """Get list of healthy providers."""
        healthy = []
        current_time = time.time()
        
        for provider in self._providers:
            name = getattr(provider, 'provider_name', str(type(provider).__name__))
            stats = self._provider_stats.get(name)
            
            if stats is None:
                healthy.append(provider)
                continue
            
            if stats.is_healthy:
                healthy.append(provider)
            elif stats.last_failure_time:
                # Check if recovery timeout has passed
                if current_time - stats.last_failure_time >= self.config.recovery_timeout:
                    stats.is_healthy = True
                    stats.consecutive_failures = 0
                    healthy.append(provider)
        
        return healthy
    
    def _select_provider(self) -> Optional[Any]:
        """Select a provider based on strategy."""
        healthy = self._get_healthy_providers()
        if not healthy:
            # All providers unhealthy, try any
            healthy = self._providers
        
        if not healthy:
            return None
        
        if self.config.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            provider = healthy[self._round_robin_index % len(healthy)]
            self._round_robin_index += 1
            return provider
        
        elif self.config.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(healthy)
        
        elif self.config.strategy == LoadBalanceStrategy.LEAST_PENDING:
            min_pending = float('inf')
            selected = healthy[0]
            for p in healthy:
                name = getattr(p, 'provider_name', str(type(p).__name__))
                stats = self._provider_stats.get(name)
                pending = stats.pending_requests if stats else 0
                if pending < min_pending:
                    min_pending = pending
                    selected = p
            return selected
        
        elif self.config.strategy == LoadBalanceStrategy.WEIGHTED:
            weights = []
            for p in healthy:
                name = getattr(p, 'provider_name', str(type(p).__name__))
                weight = self.config.weights.get(name, 1.0)
                weights.append(weight)
            return random.choices(healthy, weights=weights, k=1)[0]
        
        elif self.config.strategy == LoadBalanceStrategy.FAILOVER:
            # Use first healthy provider (primary with fallback)
            return healthy[0]
        
        return healthy[0]
    
    def _get_semaphore(self, name: str) -> asyncio.Semaphore:
        """Get or create semaphore for a provider (lazy init for event loop safety).
        
        Creates new semaphores if event loop changed (handles asyncio.run() creating new loops).
        """
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = None
        
        # Check if event loop changed - if so, clear all semaphores
        if self._semaphore_loop_id is not None and self._semaphore_loop_id != current_loop_id:
            # Event loop changed, clear all semaphores
            self._provider_semaphores.clear()
        
        self._semaphore_loop_id = current_loop_id
        
        # Create semaphore if needed
        if name not in self._provider_semaphores:
            self._provider_semaphores[name] = asyncio.Semaphore(
                self.config.max_concurrent_per_provider
            )
        
        return self._provider_semaphores[name]
    
    async def _call_provider(
        self,
        provider: Any,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Call a specific provider with tracking."""
        name = getattr(provider, 'provider_name', str(type(provider).__name__))
        stats = self._provider_stats[name]
        semaphore = self._get_semaphore(name)
        
        async with semaphore:
            stats.pending_requests += 1
            stats.total_requests += 1
            start_time = time.time()
            
            try:
                result = await provider.generate(system_prompt, messages)
                
                latency = time.time() - start_time
                stats.successful_requests += 1
                stats.total_latency += latency
                stats.consecutive_failures = 0
                
                return result
                
            except Exception as e:
                stats.failed_requests += 1
                stats.consecutive_failures += 1
                stats.last_failure_time = time.time()
                
                if stats.consecutive_failures >= self.config.failure_threshold:
                    stats.is_healthy = False
                    print(f"[LoadBalancer] Provider {name} marked unhealthy after {stats.consecutive_failures} failures")
                
                raise
                
            finally:
                stats.pending_requests -= 1
    
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Generate a response using load-balanced providers.
        
        Args:
            system_prompt: System prompt
            messages: Message list
            
        Returns:
            Generated response
        """
        # If queue is running, use it
        if self._running:
            future = await self._queue.enqueue(system_prompt, messages)
            return await future
        
        # Direct call (no queue)
        return await self._generate_direct(system_prompt, messages)
    
    async def _generate_direct(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Direct generation without queue."""
        last_error = None
        tried_providers = set()
        
        for attempt in range(len(self._providers)):
            provider = self._select_provider()
            if provider is None:
                raise RuntimeError("No providers available")
            
            name = getattr(provider, 'provider_name', str(type(provider).__name__))
            if name in tried_providers:
                continue
            tried_providers.add(name)
            
            try:
                return await self._call_provider(provider, system_prompt, messages)
            except Exception as e:
                last_error = e
                print(f"[LoadBalancer] Provider {name} failed: {e}")
                continue
        
        if last_error:
            raise last_error
        raise RuntimeError("All providers failed")
    
    async def _worker(self, worker_id: int):
        """Worker coroutine that processes queued requests."""
        while self._running:
            request = await self._queue.dequeue()
            if request is None:
                continue
            
            try:
                result = await self._generate_direct(
                    request.system_prompt,
                    request.messages,
                )
                await self._queue.complete(request, result)
            except Exception as e:
                await self._queue.fail(request, e)
    
    async def start_workers(self, num_workers: int = None):
        """Start worker tasks for processing queue.
        
        Args:
            num_workers: Number of workers (default: 2x providers)
        """
        if self._running:
            return
        
        self._running = True
        num_workers = num_workers or len(self._providers) * 2
        
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        
        print(f"[LoadBalancer] Started {num_workers} workers")
    
    async def stop_workers(self):
        """Stop all worker tasks."""
        self._running = False
        for task in self._workers:
            task.cancel()
        self._workers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        provider_stats = {}
        for name, stats in self._provider_stats.items():
            provider_stats[name] = {
                "total_requests": stats.total_requests,
                "successful": stats.successful_requests,
                "failed": stats.failed_requests,
                "success_rate": stats.success_rate,
                "avg_latency_ms": stats.avg_latency_ms,
                "pending": stats.pending_requests,
                "is_healthy": stats.is_healthy,
            }

        queue_stats = self._queue.get_stats() if self._queue else {
            "pending": 0,
            "total_enqueued": 0,
            "total_processed": 0,
            "total_timeouts": 0,
            "total_dropped": 0,
        }

        return {
            "strategy": self.config.strategy.value,
            "providers": provider_stats,
            "queue": queue_stats,
            "healthy_providers": len(self._get_healthy_providers()),
            "total_providers": len(self._providers),
        }


def create_load_balanced_provider(
    openrouter_model: str,
    aihubmix_model: str,
    openrouter_api_key: Optional[str] = None,
    aihubmix_api_key: Optional[str] = None,
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    max_concurrent_per_provider: int = 5,
) -> LoadBalancedProvider:
    """Create a load-balanced provider with OpenRouter and AIHubMix.
    
    Args:
        openrouter_model: Model for OpenRouter
        aihubmix_model: Model for AIHubMix
        openrouter_api_key: OpenRouter API key
        aihubmix_api_key: AIHubMix API key
        strategy: Load balancing strategy
        max_concurrent_per_provider: Max concurrent per provider
        
    Returns:
        Configured LoadBalancedProvider
    """
    from redblackbench.providers.openrouter_provider import OpenRouterProvider
    from redblackbench.providers.aihubmix_provider import AIHubMixProvider
    
    providers = [
        OpenRouterProvider(
            model=openrouter_model,
            api_key=openrouter_api_key,
            rate_limit_concurrent=max_concurrent_per_provider,
        ),
        AIHubMixProvider(
            model=aihubmix_model,
            api_key=aihubmix_api_key,
        ),
    ]
    
    config = LoadBalancerConfig(
        strategy=strategy,
        max_concurrent_per_provider=max_concurrent_per_provider,
    )
    
    return LoadBalancedProvider(providers, config)

