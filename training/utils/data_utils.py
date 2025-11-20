# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData
    # Optional explicit prompts aggregated per frame across objects (aligned with masks/object order)
    # Shapes when present: [T, O, K, 2] and [T, O, K]
    point_coords: torch.FloatTensor
    point_labels: torch.IntTensor

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask
    # Optional explicit prompt inputs for this object on this frame
    point_coords: Optional[torch.Tensor] = None  # [K,2] float (x,y), or None
    point_labels: Optional[torch.Tensor] = None  # [K] int32 labels, or None


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    # For prompts, we will gather per-object prompt tensors and pad to a uniform K per frame
    step_t_point_coords = [[] for _ in range(T)]
    step_t_point_labels = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))
                # Collect prompt tensors if available; placeholders will be padded later
                if getattr(obj, 'point_coords', None) is not None and getattr(obj, 'point_labels', None) is not None:
                    pc = obj.point_coords
                    pl = obj.point_labels
                    if not isinstance(pc, torch.Tensor):
                        pc = torch.as_tensor(pc, dtype=torch.float32)
                    if not isinstance(pl, torch.Tensor):
                        pl = torch.as_tensor(pl, dtype=torch.int32)
                    step_t_point_coords[t].append(pc)
                    step_t_point_labels[t].append(pl)
                else:
                    step_t_point_coords[t].append(None)
                    step_t_point_labels[t].append(None)

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    # Determine per-frame K (number of points) and pad prompts accordingly
    padded_point_coords = []  # list of [O, K, 2]
    padded_point_labels = []  # list of [O, K]
    for t in range(T):
        Ks = [pc.shape[0] for pc in step_t_point_coords[t] if pc is not None]
        K = max(Ks) if len(Ks) > 0 else 0
        O = len(step_t_point_coords[t])
        if K == 0:
            padded_point_coords.append(torch.zeros(O, 0, 2, dtype=torch.float32))
            padded_point_labels.append(torch.zeros(O, 0, dtype=torch.int32))
            continue
        coords_t = torch.zeros(O, K, 2, dtype=torch.float32)
        labels_t = torch.zeros(O, K, dtype=torch.int32)
        for oi in range(O):
            pc = step_t_point_coords[t][oi]
            pl = step_t_point_labels[t][oi]
            if pc is None or pl is None or pc.numel() == 0:
                continue
            k_i = pc.shape[0]
            if k_i >= K:
                coords_t[oi] = pc[:K]
                labels_t[oi] = pl[:K]
            else:
                coords_t[oi, :k_i] = pc
                labels_t[oi, :k_i] = pl
        padded_point_coords.append(coords_t)
        padded_point_labels.append(labels_t)
    point_coords = torch.stack(padded_point_coords, dim=0) if len(padded_point_coords) > 0 else torch.zeros(0)
    point_labels = torch.stack(padded_point_labels, dim=0) if len(padded_point_labels) > 0 else torch.zeros(0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        point_coords=point_coords,
        point_labels=point_labels,
        dict_key=dict_key,
        batch_size=[T],
    )
