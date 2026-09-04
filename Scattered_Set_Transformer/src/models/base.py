"""Model factory for scattered kinetic mechanism classification (M1-M20)."""

from __future__ import annotations

from typing import Any


MODEL_REGISTRY = {
    "settransformer": (".set_transformer", "SetTransformer"),
}


def _lazy_import_model(module_path: str, class_name: str):
    import importlib

    module = importlib.import_module(module_path, package="models")
    return getattr(module, class_name)


def get_model(name: str, in_dim: int, num_classes: int = 20, **kwargs: Any):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = MODEL_REGISTRY[name]
    model_cls = _lazy_import_model(module_path, class_name)
    return model_cls(
        in_dim=in_dim,
        dim_hidden=kwargs.get("dim_hidden", 256),
        num_classes=num_classes,
        num_heads=kwargs.get("num_heads", 8),
        num_layers=kwargs.get("num_layers", 6),
        num_inducing=kwargs.get("num_inducing", 16),
        num_pma_seeds=kwargs.get("num_pma_seeds", 1),
        dropout=kwargs.get("dropout", 0.1),
        use_feature_gating=kwargs.get("use_feature_gating", True),
        num_classifier_heads=kwargs.get("num_classifier_heads", 1),
    )
