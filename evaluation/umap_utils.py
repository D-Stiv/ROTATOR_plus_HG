"""Utilities for exporting labelled 2D and 3D UMAP route-embedding plots."""

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd


def _as_label_frame(
    labels: Union[pd.DataFrame, np.ndarray],
    label_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Normalise dataframe or array labels to a labelled dataframe."""
    if isinstance(labels, pd.DataFrame):
        selected_columns = list(label_columns) if label_columns is not None else list(labels.columns)
        if not selected_columns:
            raise ValueError("`labels` dataframe must contain at least one label column.")
        return labels.loc[:, selected_columns].copy()

    label_array = np.asarray(labels)
    if label_array.ndim == 1:
        label_array = label_array[:, None]
    if label_array.ndim != 2:
        raise ValueError("`labels` must be a dataframe or a 1D/2D array.")

    names = list(label_columns) if label_columns is not None else [
        f"label_{idx}" for idx in range(label_array.shape[1])
    ]
    if len(names) != label_array.shape[1]:
        raise ValueError("`label_columns` must have one name per label-array column.")
    return pd.DataFrame(label_array, columns=names)


def save_umap(
    embeddings: np.ndarray,
    labels: Union[pd.DataFrame, np.ndarray],
    output_dir: Union[str, Path] = "umap_plots",
    label_columns: Optional[Iterable[str]] = None,
    n_neighbors: int = 30,
    min_dist: float = 0.10,
    random_state: int = 42,
) -> None:
    """Save 2D and 3D UMAP plots for every supplied label column.

    Parameters
    ----------
    embeddings:
        Array shaped ``(n_routes, embedding_dimension)``.
    labels:
        Either a labelled dataframe, or a 2D label-code array shaped
        ``(n_routes, n_label_columns)``.  A 1D array is also accepted.
    label_columns:
        Dataframe columns to plot, or names for the columns of a label array.
    """
    try:
        import matplotlib.pyplot as plt
        import umap.umap_ as umap
    except ImportError as exc:
        raise ImportError(
            "UMAP plotting requires `umap-learn` and `matplotlib`. "
            "Install the project's requirements first."
        ) from exc

    embedding_array = np.asarray(embeddings)
    if embedding_array.ndim != 2:
        raise ValueError("`embeddings` must be a 2D array.")

    label_frame = _as_label_frame(labels, label_columns)
    if len(label_frame) != len(embedding_array):
        raise ValueError("`embeddings` and `labels` must have the same number of rows.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    projection_2d = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    ).fit_transform(embedding_array)
    projection_3d = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    ).fit_transform(embedding_array)

    for label_name in label_frame.columns:
        values = label_frame[label_name].fillna("unknown").astype(str)
        category_codes, categories = pd.factorize(values, sort=True)
        colour_map = plt.get_cmap("tab20", max(len(categories), 1))

        legend_handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="", color=colour_map(idx),
                label=str(category), markersize=5,
            )
            for idx, category in enumerate(categories)
        ]

        fig, axis = plt.subplots(figsize=(12, 9))
        axis.scatter(
            projection_2d[:, 0], projection_2d[:, 1], c=category_codes,
            cmap=colour_map, s=8, alpha=0.7, linewidths=0, rasterized=True,
        )
        axis.set(title=f"2D UMAP coloured by {label_name}", xlabel="UMAP 1", ylabel="UMAP 2")
        if len(categories) <= 20:
            axis.legend(handles=legend_handles, title=label_name,
                        bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(destination / f"umap_2d_{label_name}.png", dpi=250)
        plt.close(fig)

        fig = plt.figure(figsize=(12, 9))
        axis = fig.add_subplot(111, projection="3d")
        axis.scatter(
            projection_3d[:, 0], projection_3d[:, 1], projection_3d[:, 2],
            c=category_codes, cmap=colour_map, s=5, alpha=0.65,
            linewidths=0, rasterized=True,
        )
        axis.set(title=f"3D UMAP coloured by {label_name}",
                 xlabel="UMAP 1", ylabel="UMAP 2", zlabel="UMAP 3")
        if len(categories) <= 20:
            axis.legend(handles=legend_handles, title=label_name,
                        bbox_to_anchor=(1.15, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(destination / f"umap_3d_{label_name}.png", dpi=250)
        plt.close(fig)
