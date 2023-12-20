#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar, Union

import torch.utils.data

from . import S3Reader
from ._s3bucket_key import S3BucketKey
from ._s3client import S3Client
from ._s3dataset_common import get_objects_from_prefix, get_objects_from_uris, identity


class S3IterableDataset(torch.utils.data.IterableDataset):
    """An IterableStyle dataset created from S3 objects.

    To create an instance of S3IterableDataset, you need to use
    `from_prefix` or `from_objects` methods.
    """

    def __init__(
        self,
        region: str,
        get_dataset_objects: Callable[[S3Client], Iterable[S3BucketKey]],
        transform: Callable[[S3Reader], Any] = identity,
    ) -> None:
        self._get_dataset_objects = get_dataset_objects
        self._transform = transform
        self._region = region
        self._client: Optional[S3Client] = None

    @property
    def region(self) -> str:
        return self._region

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        transform: Callable[[S3Reader], Any] = identity,
    ) -> "S3IterableDataset":
        """Returns an instance of S3IterableDataset using the S3 URI(s) provided.

        Args:
          object_uris(str | Iterable[str]): S3 URI of the object(s) desired.
          region(str): AWS region of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3IterableDataset: An IterableStyle dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        return cls(
            region, partial(get_objects_from_uris, object_uris), transform=transform
        )

    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        transform: Callable[[S3Reader], Any] = identity,
    ) -> "S3IterableDataset":
        """Returns an instance of S3IterableDataset using the S3 URI provided.

        Args:
          s3_uri(str): An S3 URI (prefix) of the object(s) desired. Objects matching the prefix will be included in the returned dataset.
          region(str): AWS region of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3IterableDataset: An IterableStyle dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        return cls(
            region, partial(get_objects_from_prefix, s3_uri), transform=transform
        )

    def _get_client(self) -> S3Client:
        if self._client is None:
            self._client = S3Client(self.region)
        return self._client

    def _get_transformed_object(self, bucket_key: S3BucketKey) -> Any:
        return self._transform(
            self._get_client().get_object(bucket_key.bucket, bucket_key.key)
        )

    def __iter__(self) -> Iterator[Any]:
        return map(
            self._get_transformed_object, self._get_dataset_objects(self._get_client())
        )
