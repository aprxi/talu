"""Simple short-prompt scenario for POST /v1/responses."""

from scenario import Scenario, http_post_stream, extract_generation_metrics, model_uri

# Config keys that map to /v1/responses request body fields.
_API_FIELDS = {
    "max_tokens": "max_output_tokens",
    "temperature": "temperature",
    "seed": "seed",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
}


class Hello(Scenario):
    name = "responses/hello"
    description = "Short prompt, free-form reply — raw decode throughput."
    endpoint = "POST /v1/responses"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        url = f"{base_url}/v1/responses"
        model_uris: list[str] = config.get("model_uri", [])
        precisions: list[str] = config.get("precision", ["original"])
        all_results: list[dict] = []

        for base_model in model_uris:
            for scheme in precisions:
                is_original = scheme == "original"
                uri = model_uri(base_model, None if is_original else scheme)
                print(f"\n  {uri}")

                body: dict = {
                    "model": uri,
                    "input": "hello",
                    "instructions": "ok",
                    "stream": config.get("streaming", True),
                    "store": True,
                }
                for cfg_key, api_key in _API_FIELDS.items():
                    if cfg_key in config:
                        body[api_key] = config[cfg_key]

                for i in range(rounds):
                    print(f"    round {i+1}/{rounds} ...", end="", flush=True)
                    events, wall_s = http_post_stream(url, body)
                    m = extract_generation_metrics(events)
                    m["round"] = i + 1
                    m["scheme"] = scheme
                    m["model"] = base_model
                    m["model_uri"] = uri
                    m["wall_s"] = round(wall_s, 3)
                    all_results.append(m)
                    print(f" {m['engine_tok_s']} tok/s")

        return all_results
