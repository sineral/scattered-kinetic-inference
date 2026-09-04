"""Model factory for Nature mechanism classification (M1-M20)."""

from __future__ import annotations

from typing import Any


MODEL_REGISTRY = {
    "deepsets": (".deep_sets", "DeepSets"),
    "settransformer": (".set_transformer", "SetTransformer"),
    "tst": (".ts_transformer", "TimeSeriesTransformer"),
}


def _lazy_import_model(module_path: str, class_name: str):
    import importlib

    module = importlib.import_module(module_path, package="models")
    return getattr(module, class_name)


def get_model(name: str, in_dim: int = 0, num_classes: int = 20, **kwargs: Any):
    """Instantiate a model by architecture name.

    Parameters
    ----------
    name : str
        One of: settransformer, deepsets, tst.
    in_dim : int
        Feature dimension per set element (used by Set Transformer / DeepSets).
    num_classes : int
        Number of mechanism classes (20).
    """
    name = name.lower().replace("-", "_")
    if name not in MODEL_REGISTRY:
        if "deepset" in name:
            name = "deepsets"
        elif name.startswith("tst") or "timeseries" in name.replace("_", ""):
            name = "tst"
        elif "set" in name:
            name = "settransformer"
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    module_path, class_name = MODEL_REGISTRY[name]
    model_cls = _lazy_import_model(module_path, class_name)

    if name == "settransformer":
        return model_cls(
            in_dim=in_dim,
            dim_hidden=kwargs.get("dim_hidden", 256),
            num_classes=num_classes,
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 6),
            num_inducing=kwargs.get("num_inducing", 16),
            num_pma_seeds=kwargs.get("num_pma_seeds", 4),
            dropout=kwargs.get("dropout", 0.1),
            use_feature_gating=kwargs.get("use_feature_gating", True),
            num_classifier_heads=kwargs.get("num_classifier_heads", 1),
        )

    if name == "deepsets":
        return model_cls(
            in_dim=in_dim,
            dim_hidden=kwargs.get("dim_hidden", 128),
            latent_dim=kwargs.get("latent_dim", 128),
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.1),
        )

    return model_cls(
        dim_x1=kwargs.get("dim_x1", 4),
        dim_x2=kwargs.get("dim_x2", 12),
        dim_hidden=kwargs.get("dim_hidden", 128),
        num_classes=num_classes,
        num_heads=kwargs.get("num_heads", 4),
        num_layers=kwargs.get("num_layers", 3),
        dropout=kwargs.get("dropout", 0.1),
        readout=kwargs.get("readout", "attn"),
    )
