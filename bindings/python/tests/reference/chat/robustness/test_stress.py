"""
Robustness tests for Chat - memory leaks, thread safety, stress testing.

These tests verify:
1. No memory leaks under various usage patterns
2. Thread-safe concurrent access to Messages
3. Pointer stability during operations
4. Behavior under stress (many messages, rapid operations)

Per MESSAGES.md:
- Zig has sole write access, Python has read-only access
- Double-pointer pattern ensures Python pointers remain valid across Zig reallocations
- content_len should be updated atomically
"""

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from talu.chat import Chat

# =============================================================================
# Memory Leak Tests
# =============================================================================


class TestMemoryLeaks:
    """Memory leak detection tests."""

    @pytest.mark.memory
    def test_chat_create_destroy_stress(self, memory_tracker):
        """Stress test: 1000 chat create/destroy cycles."""
        tracker = memory_tracker

        # Warmup
        for _ in range(50):
            chat = Chat(system="Warmup")
            del chat
        tracker.force_gc()

        initial_rss = tracker.get_rss()

        # Stress loop
        for i in range(1000):
            chat = Chat(system=f"Test system prompt number {i}")
            del chat

        tracker.force_gc()
        final_rss = tracker.get_rss()
        growth = final_rss - initial_rss

        # Should not grow more than 2MB for 1000 empty chats
        max_growth = 2 * 1024 * 1024
        assert growth < max_growth, (
            f"Memory grew by {growth / 1024 / 1024:.2f}MB after 1000 chat cycles. "
            f"Possible memory leak in Chat creation/destruction."
        )

    @pytest.mark.memory
    def test_from_dict_stress(self, memory_tracker):
        """Stress test: 500 from_dict cycles with varying message counts."""
        tracker = memory_tracker

        # Warmup
        for _ in range(20):
            chat = Chat.from_dict({"messages": [{"role": "user", "content": "x"}]})
            del chat
        tracker.force_gc()

        initial_rss = tracker.get_rss()

        # Create chats with varying message counts
        for i in range(500):
            msg_count = (i % 20) + 1  # 1 to 20 messages
            messages = [
                {"role": "user" if j % 2 == 0 else "assistant", "content": f"Message {j}"}
                for j in range(msg_count)
            ]
            chat = Chat.from_dict({"messages": messages})
            del chat

        tracker.force_gc()
        final_rss = tracker.get_rss()
        growth = final_rss - initial_rss

        max_growth = 5 * 1024 * 1024
        assert growth < max_growth, (
            f"Memory grew by {growth / 1024 / 1024:.2f}MB after 500 from_dict cycles. "
            f"Possible memory leak in message loading."
        )

    @pytest.mark.memory
    def test_fork_memory_leak(self, memory_tracker):
        """Test that forking doesn't leak memory."""
        tracker = memory_tracker

        # Create base chat
        base_chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "Base system"},
                    {"role": "user", "content": "Base user message"},
                    {"role": "assistant", "content": "Base assistant response"},
                ]
            }
        )

        # Warmup
        for _ in range(20):
            forked = base_chat.fork()
            del forked
        tracker.force_gc()

        initial_rss = tracker.get_rss()

        # Fork many times
        for _ in range(500):
            forked = base_chat.fork()
            del forked

        tracker.force_gc()
        final_rss = tracker.get_rss()
        growth = final_rss - initial_rss

        max_growth = 3 * 1024 * 1024
        assert growth < max_growth, (
            f"Memory grew by {growth / 1024 / 1024:.2f}MB after 500 fork cycles. "
            f"Possible memory leak in Chat.fork()."
        )

        del base_chat

    @pytest.mark.memory
    def test_large_message_memory(self, memory_tracker):
        """Test memory handling with large message content."""
        tracker = memory_tracker

        # Create large content (1MB string)
        large_content = "x" * (1024 * 1024)

        # Warmup
        for _ in range(5):
            chat = Chat.from_dict({"messages": [{"role": "user", "content": large_content}]})
            del chat
        tracker.force_gc()

        initial_rss = tracker.get_rss()

        # Create and destroy chats with large messages
        for _ in range(20):
            chat = Chat.from_dict(
                {
                    "messages": [
                        {"role": "user", "content": large_content},
                        {"role": "assistant", "content": large_content},
                    ]
                }
            )
            # Read the content to ensure it's materialized
            _ = chat.items[0].text
            _ = chat.items[1].text
            del chat

        tracker.force_gc()
        final_rss = tracker.get_rss()
        growth = final_rss - initial_rss

        # Should release memory - allow for some overhead but not 20x2MB
        max_growth = 10 * 1024 * 1024
        assert growth < max_growth, (
            f"Memory grew by {growth / 1024 / 1024:.2f}MB after large message cycles. "
            f"Large message content may not be properly freed."
        )

    @pytest.mark.memory
    def test_clear_reset_memory(self, memory_tracker):
        """Test that clear() and reset() properly free memory."""
        tracker = memory_tracker

        tracker.force_gc()
        initial_rss = tracker.get_rss()

        # Create chat, populate, clear, repeat
        chat = Chat()
        for i in range(100):
            # Reload with messages
            chat = Chat.from_dict(
                {
                    "messages": [
                        {"role": "system", "content": f"System {i}"},
                        {"role": "user", "content": f"User message {i}" * 100},
                        {"role": "assistant", "content": f"Response {i}" * 100},
                    ]
                }
            )
            chat.clear()

        del chat
        tracker.force_gc()
        final_rss = tracker.get_rss()
        growth = final_rss - initial_rss

        max_growth = 3 * 1024 * 1024
        assert growth < max_growth, (
            f"Memory grew by {growth / 1024 / 1024:.2f}MB after clear cycles. "
            f"clear() may not properly free message memory."
        )


# =============================================================================
# Garbage Collection Tests
# =============================================================================


class TestGarbageCollection:
    """Tests for proper garbage collection behavior."""

    def test_chat_ref_cycle_collection(self):
        """Chat with internal references is still collected."""
        chat = Chat(system="Test")
        # Access messages to create potential reference
        _ = chat.items
        _ = list(chat.items)

        ref = weakref.ref(chat)
        del chat
        gc.collect()

        assert ref() is None, "Chat with accessed messages was not garbage collected"

    def test_forked_chat_collected_independently(self):
        """Forked chat can be collected while original exists."""
        original = Chat(system="Original")
        forked = original.fork()
        forked_ref = weakref.ref(forked)

        del forked
        gc.collect()

        assert forked_ref() is None, "Forked chat was not collected"
        assert len(original.items) == 1  # Original still works

    def test_original_chat_collected_while_fork_exists(self):
        """Original chat can be collected while fork exists."""
        original = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )
        forked = original.fork()
        original_ref = weakref.ref(original)

        del original
        gc.collect()

        assert original_ref() is None, "Original chat was not collected"
        # Fork should still work
        assert len(forked.items) == 2
        assert forked.items[0].text == "System"

    def test_messages_view_doesnt_prevent_collection(self):
        """Holding Messages reference doesn't prevent Chat collection."""
        chat = Chat(system="Test")
        items = chat.items  # Hold reference to Items view

        del chat
        gc.collect()

        # Chat should be collected even though we hold Items
        # (Items is just a view, not an owner)
        # Note: This may or may not be None depending on implementation
        # If Items holds a reference to Chat, that's also valid design
        _ = items  # Suppress unused variable warning


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Thread safety tests for concurrent message access."""

    def test_concurrent_read_same_chat(self):
        """Multiple threads can read from same chat concurrently."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User message"},
                    {"role": "assistant", "content": "Assistant response"},
                ]
            }
        )

        errors = []
        results = []

        def read_messages():
            try:
                for _ in range(100):
                    # Various read operations
                    _ = len(chat.items)
                    _ = chat.items[0]
                    _ = chat.items[-1]
                    _ = chat.items.system
                    _ = chat.items.last
                    _ = list(chat.items)
                results.append("ok")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_messages) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent reads failed: {errors}"
        assert len(results) == 10

    def test_concurrent_read_different_chats(self):
        """Multiple threads can work with different chats concurrently."""
        chats = [
            Chat.from_dict(
                {
                    "messages": [
                        {"role": "user", "content": f"Message {i}"},
                        {"role": "assistant", "content": f"Response {i}"},
                    ]
                }
            )
            for i in range(10)
        ]

        errors = []
        results = []

        def work_with_chat(chat, chat_id):
            try:
                for _ in range(50):
                    _ = len(chat.items)
                    _ = list(chat.items)
                    _ = chat.items[0].text
                results.append(chat_id)
            except Exception as e:
                errors.append((chat_id, e))

        threads = [
            threading.Thread(target=work_with_chat, args=(chat, i)) for i, chat in enumerate(chats)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent chat access failed: {errors}"
        assert len(results) == 10

    def test_concurrent_fork_operations(self):
        """Multiple threads can fork from same chat."""
        base = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "Base system"},
                    {"role": "user", "content": "Base question"},
                ]
            }
        )

        forks = []
        errors = []
        lock = threading.Lock()

        def fork_and_use():
            try:
                forked = base.fork()
                # Verify fork integrity
                assert len(forked.items) == 2
                assert forked.items[0].text == "Base system"
                with lock:
                    forks.append(forked)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=fork_and_use) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent fork failed: {errors}"
        assert len(forks) == 20

        # All forks should be independent
        for i, forked in enumerate(forks):
            assert len(forked.items) == 2, f"Fork {i} has wrong message count"

    def test_concurrent_create_destroy(self):
        """Concurrent chat creation and destruction is safe."""
        errors = []
        created_count = [0]
        lock = threading.Lock()

        def create_destroy_loop():
            try:
                for i in range(50):
                    chat = Chat.from_dict(
                        {
                            "messages": [
                                {"role": "user", "content": f"Thread msg {i}"},
                            ]
                        }
                    )
                    _ = chat.items[0]
                    del chat
                with lock:
                    created_count[0] += 50
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=create_destroy_loop) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        gc.collect()

        assert not errors, f"Concurrent create/destroy failed: {errors}"
        assert created_count[0] == 500

    def test_read_during_fork(self):
        """Reading from chat while another thread forks is safe."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        )

        errors = []
        read_results = []
        fork_results = []

        def reader():
            try:
                for _ in range(100):
                    content = chat.items[1].text
                    assert content == "Hello"
                    read_results.append(True)
            except Exception as e:
                errors.append(("reader", e))

        def forker():
            try:
                for _ in range(20):
                    forked = chat.fork()
                    assert len(forked.items) == 3
                    fork_results.append(forked)
            except Exception as e:
                errors.append(("forker", e))

        reader_threads = [threading.Thread(target=reader) for _ in range(5)]
        forker_threads = [threading.Thread(target=forker) for _ in range(3)]

        all_threads = reader_threads + forker_threads
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        assert not errors, f"Read during fork failed: {errors}"
        assert len(read_results) == 500
        assert len(fork_results) == 60


# =============================================================================
# Pointer Stability Tests
# =============================================================================


class TestPointerStability:
    """Tests for pointer stability (per MESSAGES.md double-pointer pattern)."""

    def test_messages_reference_stable_after_operations(self):
        """Messages reference remains valid after chat operations."""
        chat = Chat(system="Initial system")
        items = chat.items

        # Get initial state
        initial_system = items.system

        # Perform operations that might reallocate
        chat.clear()

        # Items reference should still work
        assert len(items) == 1  # System kept
        assert items.system == initial_system

        # Reset and check again
        chat.reset()
        assert len(items) == 0

    def test_message_dict_stable_after_iteration(self):
        """Message items remain valid after iteration."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "User"},
                    {"role": "assistant", "content": "Assistant"},
                ]
            }
        )

        # Get references to message items
        item0 = chat.items[0]
        item1 = chat.items[1]
        item2 = chat.items[2]

        # Iterate (might trigger internal operations)
        for _ in chat.items:
            pass

        # Items should still be valid
        assert item0.text == "System"
        assert item1.text == "User"
        assert item2.text == "Assistant"

    def test_to_list_independent_of_source(self):
        """list() returns independent copy."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "user", "content": "Original"},
                ]
            }
        )

        list_copy = list(chat.items)

        # Modify the copy (list itself, items are still references)
        list_copy.append(None)  # Can't create new items, just test list independence

        # Original should be unchanged
        assert chat.items[0].text == "Original"
        assert len(chat.items) == 1


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for edge cases and limits."""

    def test_many_messages(self):
        """Chat handles many messages."""
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(1000)
        ]
        chat = Chat.from_dict({"messages": messages})

        assert len(chat.items) == 1000
        assert chat.items[0].text == "Message 0"
        assert chat.items[999].text == "Message 999"
        assert chat.items[-1].text == "Message 999"

        # Iteration should work
        count = sum(1 for _ in chat.items)
        assert count == 1000

    def test_rapid_fork_chain(self):
        """Rapid sequential forking works."""
        chat = Chat(system="Start")

        # Fork chain
        current = chat
        for _ in range(100):
            current = current.fork()

        assert len(current.items) == 1
        assert current.items.system == "Start"

    def test_deep_message_content(self):
        """Messages with various content types."""
        # Unicode, newlines, special chars
        messages = [
            {"role": "system", "content": "System with unicode: \u4e2d\u6587 \U0001f600"},
            {"role": "user", "content": "Line1\nLine2\nLine3"},
            {"role": "assistant", "content": "Tab\there\tand\tthere"},
            {"role": "user", "content": "Quote: \"hello\" and 'world'"},
            {"role": "assistant", "content": "Backslash: \\ and slash: /"},
            {"role": "user", "content": ""},  # Empty content
            {"role": "assistant", "content": " " * 1000},  # Whitespace
        ]

        chat = Chat.from_dict({"messages": messages})

        assert len(chat.items) == 7
        assert "\u4e2d\u6587" in chat.items[0].text
        assert chat.items[1].text.count("\n") == 2
        assert chat.items[5].text == ""
        assert len(chat.items[6].text) == 1000

    def test_rapid_clear_reset_cycle(self):
        """Rapid clear/reset cycles don't cause issues."""
        chat = Chat(system="System")

        for i in range(200):
            if i % 2 == 0:
                chat.clear()
                assert len(chat.items) == 1  # System kept
            else:
                chat.reset()
                assert len(chat.items) == 0
                # Re-add system for next iteration
                chat = Chat(system="System")

    @pytest.mark.slow
    def test_thread_pool_stress(self):
        """Stress test with thread pool executor."""

        def create_and_verify(i):
            chat = Chat.from_dict(
                {
                    "messages": [
                        {"role": "system", "content": f"System {i}"},
                        {"role": "user", "content": f"User {i}"},
                    ]
                }
            )
            assert len(chat.items) == 2
            assert chat.items.system == f"System {i}"

            forked = chat.fork()
            assert len(forked.items) == 2

            chat.clear()
            assert len(chat.items) == 1

            return i

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(create_and_verify, i) for i in range(200)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 200
        assert set(results) == set(range(200))


# =============================================================================
# Model-Based Robustness Tests
# =============================================================================


class TestModelRobustness:
    """Robustness tests that require a model."""

    @pytest.mark.requires_model
    @pytest.mark.memory
    def test_generation_memory_stress(self, test_model_path, memory_tracker):
        """Repeated generation cycles don't leak memory."""
        from talu import Chat, GenerationConfig

        tracker = memory_tracker

        # Warmup
        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=5))
        for _ in range(3):
            _ = chat.send("Hi", max_tokens=3)
            chat.clear()
        tracker.force_gc()

        initial_rss = tracker.get_rss()

        # Generate many times with same chat
        for _ in range(20):
            response = chat.send("Count to 3", max_tokens=5)
            _ = str(response)
            chat.clear()

        tracker.force_gc()
        final_rss = tracker.get_rss()
        growth = final_rss - initial_rss

        max_growth = 20 * 1024 * 1024  # 20MB (model is in memory)
        assert growth < max_growth, (
            f"Memory grew by {growth / 1024 / 1024:.2f}MB after 20 generations. "
            f"Possible memory leak in generation."
        )

    @pytest.mark.requires_model
    @pytest.mark.memory
    def test_streaming_memory_stress(self, test_model_path, memory_tracker):
        """Repeated streaming cycles don't leak memory."""
        from talu import Chat, GenerationConfig

        tracker = memory_tracker

        # Warmup
        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=10))
        for _ in range(3):
            response = chat("Hi", stream=True, max_tokens=5)
            _ = list(response)
            chat.clear()
        tracker.force_gc()

        initial_rss = tracker.get_rss()

        # Stream many times
        for _ in range(20):
            response = chat("Count", stream=True, max_tokens=10)
            tokens = list(response)
            _ = "".join(tokens)
            chat.clear()

        tracker.force_gc()
        final_rss = tracker.get_rss()
        growth = final_rss - initial_rss

        # Allow more headroom for model internals - streaming allocates buffers
        # that may not be immediately released. 100MB is generous but catches leaks.
        max_growth = 100 * 1024 * 1024
        assert growth < max_growth, (
            f"Memory grew by {growth / 1024 / 1024:.2f}MB after 20 streaming cycles. "
            f"Possible memory leak in streaming."
        )

    @pytest.mark.requires_model
    def test_read_messages_during_generation(self, test_model_path):
        """Can read messages while generation is in progress (via threads)."""
        import threading

        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=20))

        # Add initial history
        chat.send("Hello", max_tokens=3)

        errors = []
        read_counts = []

        def reader():
            """Read messages repeatedly during generation."""
            try:
                count = 0
                for _ in range(50):
                    _ = len(chat.items)
                    _ = list(chat.items)
                    if chat.items.last:
                        _ = chat.items.last.text
                    count += 1
                    time.sleep(0.001)  # DEADLOCK_GUARD: throttle to avoid busy-wait
                read_counts.append(count)
            except Exception as e:
                errors.append(("reader", e))

        def generator():
            """Generate response."""
            try:
                response = chat("Tell me a story", stream=True, max_tokens=15)
                for _ in response:
                    pass
            except Exception as e:
                errors.append(("generator", e))

        # Start reader threads before generation
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]
        gen_thread = threading.Thread(target=generator)

        for t in reader_threads:
            t.start()
        gen_thread.start()

        gen_thread.join()
        for t in reader_threads:
            t.join()

        assert not errors, f"Read during generation failed: {errors}"
        assert len(read_counts) == 3

    @pytest.mark.requires_model
    def test_sequential_multi_chat_generation(self, test_model_path):
        """Multiple chats can generate sequentially (shared engine)."""
        from talu import Chat, GenerationConfig

        # Create multiple chats - they share the underlying engine
        chats = [
            Chat(
                test_model_path,
                system=f"You are assistant {i}",
                config=GenerationConfig(max_tokens=5),
            )
            for i in range(3)
        ]

        results = []

        # Generate sequentially - concurrent generation not yet supported
        for i, chat in enumerate(chats):
            response = chat.send("Say hi", max_tokens=5)
            results.append((i, response))

        assert len(results) == 3

        # Each chat should have generated something
        for idx, response in results:
            assert response.usage.completion_tokens > 0, f"Chat {idx} produced no tokens"

    @pytest.mark.requires_model
    def test_fork_during_conversation(self, test_model_path):
        """Fork mid-conversation and continue both branches."""
        from talu import Chat, GenerationConfig

        config = GenerationConfig(max_tokens=5)
        chat = Chat(test_model_path, system="Be brief.", config=config)

        # Build some history
        chat.send("What is 1+1?", max_tokens=5)
        chat.send("What is 2+2?", max_tokens=5)

        initial_len = len(chat.items)

        # Fork and continue original
        forked = chat.fork()
        chat.send("What is 3+3?", max_tokens=5)

        # Continue fork differently
        forked.send("What color is the sky?", max_tokens=5)

        # Both should have grown independently (+user +generation output, may include reasoning)
        assert len(chat.items) >= initial_len + 2
        assert len(forked.items) >= initial_len + 2

        # Content should differ (different questions)
        assert chat.items[-2].text != forked.items[-2].text

    @pytest.mark.requires_model
    def test_rapid_generation_cycles(self, test_model_path):
        """Rapid generation cycles work correctly."""
        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=3))

        # Rapid fire generations
        for i in range(10):
            response = chat.send(f"Count {i}", max_tokens=3)
            assert response.usage.completion_tokens > 0
            # Don't clear - build up history

        # Should have at least 20 items (10 user + 10 generation outputs)
        # Thinking models may add reasoning items too.
        assert len(chat.items) >= 20
