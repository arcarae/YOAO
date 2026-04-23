"""Response cache for LLM API calls.

Provides caching to avoid redundant API calls for identical prompts.
Useful for deterministic scenarios or repeated system prompts.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import threading


@dataclass
class CacheConfig:
    """Configuration for response caching.
    
    Attributes:
        enabled: Whether caching is enabled
        max_entries: Maximum number of cached entries
        ttl_seconds: Time-to-live for cache entries (0 = no expiration)
        cache_by_temperature: Include temperature in cache key
        min_prompt_length: Minimum prompt length to cache (avoid caching trivial prompts)
    """
    enabled: bool = True
    max_entries: int = 1000
    ttl_seconds: float = 3600.0  # 1 hour default
    cache_by_temperature: bool = True
    min_prompt_length: int = 50


@dataclass
class CacheEntry:
    """A cached response entry."""
    response: str
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """LRU cache for LLM responses with TTL support.
    
    Thread-safe and async-compatible.
    """
    
    def __init__(self, config: CacheConfig):
        """Initialize the response cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _make_key(
        self,
        model: str,
        system_prompt: str,
        messages: List[dict],
        temperature: float = 0.0,
    ) -> str:
        """Create a cache key from request parameters.
        
        Args:
            model: Model identifier
            system_prompt: System prompt
            messages: List of message dicts
            temperature: Sampling temperature
            
        Returns:
            SHA256 hash of the parameters
        """
        key_data = {
            "model": model,
            "system": system_prompt,
            "messages": messages,
        }
        if self.config.cache_by_temperature:
            key_data["temperature"] = temperature
        
        # Create deterministic JSON string
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        model: str,
        system_prompt: str,
        messages: List[dict],
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Get a cached response if available.
        
        Args:
            model: Model identifier
            system_prompt: System prompt
            messages: List of message dicts
            temperature: Sampling temperature
            
        Returns:
            Cached response or None if not found/expired
        """
        if not self.config.enabled:
            return None
        
        # Skip caching for short prompts
        total_length = len(system_prompt) + sum(
            len(m.get("content", "")) for m in messages
        )
        if total_length < self.config.min_prompt_length:
            return None
        
        key = self._make_key(model, system_prompt, messages, temperature)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if self.config.ttl_seconds > 0:
                if time.time() - entry.created_at > self.config.ttl_seconds:
                    del self._cache[key]
                    self._misses += 1
                    return None
            
            # Update access tracking (LRU)
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            self._hits += 1
            return entry.response
    
    async def get_async(
        self,
        model: str,
        system_prompt: str,
        messages: List[dict],
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Async version of get()."""
        # For simple dict operations, sync lock is fine
        return self.get(model, system_prompt, messages, temperature)
    
    def put(
        self,
        model: str,
        system_prompt: str,
        messages: List[dict],
        temperature: float,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store a response in the cache.
        
        Args:
            model: Model identifier
            system_prompt: System prompt
            messages: List of message dicts
            temperature: Sampling temperature
            response: Response to cache
            metadata: Optional metadata to store
        """
        if not self.config.enabled:
            return
        
        # Skip caching for short prompts
        total_length = len(system_prompt) + sum(
            len(m.get("content", "")) for m in messages
        )
        if total_length < self.config.min_prompt_length:
            return
        
        # Skip caching empty responses
        if not response or not response.strip():
            return
        
        key = self._make_key(model, system_prompt, messages, temperature)
        
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.config.max_entries:
                # Remove oldest (first) entry
                self._cache.popitem(last=False)
                self._evictions += 1
            
            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
                metadata=metadata or {},
            )
    
    async def put_async(
        self,
        model: str,
        system_prompt: str,
        messages: List[dict],
        temperature: float,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Async version of put()."""
        self.put(model, system_prompt, messages, temperature, response, metadata)
    
    def invalidate(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Invalidate cache entries matching criteria.
        
        Args:
            model: If provided, only invalidate entries for this model
            system_prompt: If provided, only invalidate entries with this prompt
        """
        if model is None and system_prompt is None:
            self.clear()
            return
        
        with self._lock:
            # This is inefficient for large caches, but provides flexibility
            # In practice, we'd need to index by model/prompt for fast invalidation
            keys_to_remove = []
            for key, entry in self._cache.items():
                # Check metadata for matching criteria
                if model and entry.metadata.get("model") != model:
                    continue
                if system_prompt and entry.metadata.get("system_prompt") != system_prompt:
                    continue
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "enabled": self.config.enabled,
                "entries": len(self._cache),
                "max_entries": self.config.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        if self.config.ttl_seconds <= 0:
            return 0
        
        removed = 0
        now = time.time()
        
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if now - entry.created_at > self.config.ttl_seconds
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                removed += 1
        
        return removed


class PromptHasher:
    """Utility for creating stable hashes of prompts.
    
    Used for prompt caching with providers that support it (e.g., Anthropic).
    """
    
    @staticmethod
    def hash_system_prompt(system_prompt: str) -> str:
        """Create a hash of the system prompt for caching purposes.
        
        Args:
            system_prompt: The system prompt
            
        Returns:
            Short hash string
        """
        return hashlib.md5(system_prompt.encode()).hexdigest()[:16]
    
    @staticmethod
    def get_cacheable_prefix(
        system_prompt: str,
        messages: List[dict],
        max_cacheable_messages: int = 5,
    ) -> Tuple[str, List[dict], List[dict]]:
        """Split messages into cacheable prefix and dynamic suffix.
        
        For providers that support prompt caching, the prefix can be cached.
        
        Args:
            system_prompt: System prompt
            messages: All messages
            max_cacheable_messages: Max messages to include in cacheable prefix
            
        Returns:
            Tuple of (system_prompt, cacheable_messages, remaining_messages)
        """
        if len(messages) <= max_cacheable_messages:
            return system_prompt, messages, []
        
        return (
            system_prompt,
            messages[:max_cacheable_messages],
            messages[max_cacheable_messages:],
        )


# Global cache instance (singleton pattern)
_global_cache: Optional[ResponseCache] = None
_cache_lock = threading.Lock()


def get_global_cache(config: Optional[CacheConfig] = None) -> ResponseCache:
    """Get or create the global response cache.
    
    Args:
        config: Optional config (only used if creating new cache)
        
    Returns:
        Global ResponseCache instance
    """
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = ResponseCache(config or CacheConfig())
        return _global_cache


def reset_global_cache():
    """Reset the global cache (mainly for testing)."""
    global _global_cache
    with _cache_lock:
        if _global_cache:
            _global_cache.clear()
        _global_cache = None

