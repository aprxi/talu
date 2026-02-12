"""Chat unit tests (fast, no model inference required).

Maps to: talu/chat/

Organized by functional area:
- session/: Session state, lifecycle, error handling
- config/: GenerationConfig, SamplingParams, client args
- schema/: Schema generation, hydration, conversion
- client/: Client construction
- hooks/: Streaming hooks
- api/: Convenience functions

See tests/reference/chat/ for model inference tests.
See tests/router/ for Router, ModelSpec, and backend tests.
"""
