from pathlib import Path

from src.utils.paths import PARQUET_FILE

try:
    import huggingface_hub
except ImportError:  # pragma: no cover - optional dependency
    huggingface_hub = None

try:
    import datasets
except ImportError:  # pragma: no cover - optional dependency
    datasets = None


class HuggingFaceDatasets:
    """
    Bridge between the Parquet dataset and Hugging Face Hub.

    The pipeline's storage layer writes a single Parquet file.  This
    class exposes two operations:

    * ``get_dataset`` reads the Parquet file into a
      ``datasets.Dataset``, downloads it from a Hub repo, or saves it
      locally.
    * ``push_dataset`` uploads a DataFrame/objects to a Hub repo so
      other instances of the refinery (or downstream training jobs)
      can fetch the exact same records without re-running the
      pipeline.

    The class is deliberately thin: it does not try to replace the
    existing storage layer, only to connect it to the Hugging Face
    ecosystem.  It also respects the fact that pushing to Hub
    requires credentials, so every public method accepts a
    ``use_auth_token`` argument that falls back to the standard
    ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variables.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        local_dir: str | None = None,
        token: str | None = None,
    ):
        self.repo_id = repo_id or "default/user-refinery-dataset"
        self.local_dir = Path(local_dir) if local_dir is not None else None
        self._token = token

    @property
    def token(self) -> str | None:
        if self._token is not None:
            return self._token
        token_path = Path.home().joinpath(
            ".cache", "huggingface", "token"
        )
        if token_path.exists():
            return token_path.read_text().strip()
        return None

    def _ensure_token(self, use_auth_token: str | None) -> str | None:
        """
        Resolve the authentication token for a Hub operation.

        Priority: explicit argument > environment variable > cache file
        > None (read-only / local use).
        """
        if use_auth_token:
            return use_auth_token
        if "HF_TOKEN" in Path().env or "HUGGING_FACE_HUB_TOKEN" in Path().env:
            return (
                Path().env["HF_TOKEN"]
                or Path().env["HUGGING_FACE_HUB_TOKEN"]
            )
        if self.token is not None:
            return self.token
        return None

    def get_dataset(
        self,
        split: str = "train",
        use_auth_token: str | None = None,
        **hf_kwargs,
    ):
        """
        Load the refined Parquet dataset as a HuggingFace
        ``Dataset``.

        If the dataset is already on the Hub at ``self.repo_id`` the
        file is downloaded transparently.  Otherwise the local Parquet
        file is used and, when ``local_dir`` is set, cached so future
        calls skip the I/O.  The method never pushes anything by
        itself; that is the job of :meth:`push_dataset`.

        Returns:
            A ``datasets.Dataset`` when the optional dependencies are
            installed, otherwise the path to the Parquet file so the
            pipeline stays importable in minimal environments.

        Raises:
            RuntimeError: if the ``datasets`` package is required by
                the caller but not installed and a local path is not
                available.
        """
        if huggingface_hub is None:
            # Fall back to the raw Parquet file so the pipeline
            # remains importable even without the optional deps.
            if datasets is not None:
                return datasets.Dataset.from_parquet(
                    str(PARQUET_FILE), split=split
                )
            return PARQUET_FILE

        token = self._ensure_token(use_auth_token)
        download_kwargs = {"token": token} if token else {}

        try:
            # Try to fetch from the Hub first: this is the case where
            # the refinery is run against a shared dataset.
            return huggingface_hub.hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{split}.parquet",
                repo_type="dataset",
                **download_kwargs,
            )
        except huggingface_hub.HFHubNotFoundError:
            # Not on Hub: fall back to the generated Parquet file.
            if self.local_dir is not None:
                cache_file = Path(self.local_dir) / f"{split}.parquet"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(PARQUET_FILE.read_bytes())
                return huggingface_hub.load_from_disk(str(cache_file))
            return huggingface_hub.load_from_disk(
                str(PARQUET_FILE), split=split
            )

    def push_dataset(
        self,
        dataset,
        *,
        use_auth_token: str | None = None,
        **push_kwargs,
    ) -> str:
        """
        Upload a dataset object (or a list of documents) to a Hub
        repo as a Parquet file.

        The caller owns the dataset creation; this method handles the
        upload and returns the revision that was pushed.  It is
        idempotent: re-pushing the same data with the same commit
        message yields the same revision hash.

        Args:
            dataset: Any object convertible to a HuggingFace
                ``Dataset``: a ``datasets.Dataset``, a list of
                dicts, a pandas DataFrame, or a path to parquet.

        Returns:
            The commit hash / revision of the pushed dataset on Hub.

        Raises:
            RuntimeError: if the optional dependencies are missing and
                the dataset cannot be serialised.
        """
        if huggingface_hub is None:
            raise RuntimeError(
                "huggingface_hub is required for push_dataset."
                "Install with: pip install huggingface_hub"
            )

        token = self._ensure_token(use_auth_token)
        if token:
            huggingface_hub.login(token=token)

        # Normalise the payload to a HuggingFace Dataset so the Hub
        # client can serialise it.
        if not hasattr(dataset, "to_parquet"):
            if datasets is None:
                raise RuntimeError(
                    "The datasets package is required for push_dataset."
                )
            dataset = datasets.Dataset.from_list(dataset)

        # Parquet is the lingua franca of this project and lets the Hub
        # serve the data with the same readers used by the pipeline.
        import uuid

        temp_parquet = Path("/tmp") / f"hf_refinery_{uuid.uuid4().hex}.parquet"
        dataset.to_parquet(str(temp_parquet))

        try:
            return huggingface_hub.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                exist_ok=True,
            )
        finally:
            temp_parquet.unlink(missing_ok=True)
