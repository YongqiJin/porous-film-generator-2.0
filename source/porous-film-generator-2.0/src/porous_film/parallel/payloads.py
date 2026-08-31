from __future__ import annotations

from typing import Any

from porous_film.config import GeneratorConfig


def canonical_config_payload(config: GeneratorConfig) -> dict[str, Any]:
    """Return a JSON payload that preserves the user's film input mode."""
    payload = config.model_dump(mode="json")
    film_payload = dict(payload["film"])
    if config.source_schema_version >= 3:
        film_payload.pop("packing_box_A", None)
        film_payload.pop("z_padding_A", None)
    elif config.film.z_padding_A is not None:
        film_payload.pop("packing_box_A", None)
    else:
        film_payload.pop("z_padding_A", None)
    payload["film"] = film_payload
    return payload
