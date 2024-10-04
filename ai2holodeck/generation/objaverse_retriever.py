import os
from typing import Tuple, Sequence, Optional, Callable

import compress_json
import compress_pickle
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import tqdm
from filelock import FileLock
from objathor.dataset.generate_holodeck_features import (
    ObjectDatasetDirs,
    DEFAULT_DEVICE,
)
from torch import nn
from torch.utils.data import DataLoader

from ai2holodeck.constants import (
    OBJATHOR_ANNOTATIONS_PATH,
    HOLODECK_THOR_ANNOTATIONS_PATH,
    OBJATHOR_FEATURES_DIR,
    HOLODECK_THOR_FEATURES_DIR,
)
from ai2holodeck.generation.utils import get_bbox_dims_vec


class ObjathorRetriever:
    def __init__(
        self,
        clip_model: nn.Module,
        clip_preprocess: Callable,
        clip_tokenizer: Callable,
        sbert_model: nn.Module,
        retrieval_threshold: float,
        blender_thor_similarity_threshold: float = 0.6,
    ):
        objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
        thor_annotations = compress_json.load(HOLODECK_THOR_ANNOTATIONS_PATH)

        self.database = {**objathor_annotations, **thor_annotations}

        objathor_clip_features_dict = self._load_clip_features_and_check_has_text(
            os.path.join(OBJATHOR_FEATURES_DIR, f"clip_features.pkl")
        )
        objathor_sbert_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, f"sbert_features.pkl")
        )
        assert (
            objathor_clip_features_dict["uids"] == objathor_sbert_features_dict["uids"]
        )

        objathor_uids = objathor_clip_features_dict["uids"]
        objathor_clip_img_features = objathor_clip_features_dict["img_features"].astype(
            np.float32
        )
        objathor_clip_text_features = objathor_clip_features_dict[
            "text_features"
        ].astype(np.float32)
        objathor_sbert_text_features = objathor_sbert_features_dict[
            "text_features"
        ].astype(np.float32)

        thor_clip_features_dict = self._load_clip_features_and_check_has_text(
            os.path.join(HOLODECK_THOR_FEATURES_DIR, "clip_features.pkl")
        )
        thor_sbert_features_dict = compress_pickle.load(
            os.path.join(HOLODECK_THOR_FEATURES_DIR, "sbert_features.pkl")
        )
        assert thor_clip_features_dict["uids"] == thor_sbert_features_dict["uids"]

        thor_uids = thor_clip_features_dict["uids"]
        thor_clip_img_features = thor_clip_features_dict["img_features"].astype(
            np.float32
        )
        thor_clip_text_features = thor_clip_features_dict["text_features"].astype(
            np.float32
        )
        thor_sbert_text_features = thor_sbert_features_dict["text_features"].astype(
            np.float32
        )

        self.clip_img_features = torch.from_numpy(
            np.concatenate([objathor_clip_img_features, thor_clip_img_features], axis=0)
        )
        self.clip_img_features = F.normalize(self.clip_img_features, p=2, dim=-1)

        self.clip_text_features = torch.from_numpy(
            np.concatenate(
                [objathor_clip_text_features, thor_clip_text_features], axis=0
            )
        )
        self.clip_text_features = F.normalize(self.clip_text_features, p=2, dim=-1)

        self.sbert_text_features = torch.from_numpy(
            np.concatenate(
                [objathor_sbert_text_features, thor_sbert_text_features], axis=0
            )
        )

        self.asset_ids = objathor_uids + thor_uids

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model

        self.retrieval_threshold = retrieval_threshold
        self.blender_thor_similarity_threshold = blender_thor_similarity_threshold

    def _load_clip_features_and_check_has_text(self, clip_features_path: str):
        with FileLock(clip_features_path + ".lock"):
            clip_features_dict = compress_pickle.load(clip_features_path)
            if "text_features" in clip_features_dict:
                return clip_features_dict

            print(
                f"Clip features at {clip_features_path} do not have text features."
                f" Will attempt to generate these, but this may take a while."
            )

            annotations = {
                uid: self.database[uid] for uid in clip_features_dict["uids"]
            }
            for ann in annotations.values():
                assert (
                    ann["description"] is not None
                    or ann["description_auto"] is not None
                )

            dataset = ObjectDatasetDirs(
                annotations=annotations,
                asset_dir=None,
                image_preprocessor=None,
            )

            dataloader = DataLoader(
                dataset, batch_size=16, shuffle=False, num_workers=4
            )

            clip_model_name = "ViT-L-14"
            pretrained = "laion2b_s32b_b82k"
            clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name=clip_model_name, pretrained=pretrained, device=DEFAULT_DEVICE
            )
            clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

            uids = clip_features_dict["uids"]
            new_uids = []
            clip_text_features = []
            with torch.no_grad():
                with tqdm.tqdm(total=len(uids)) as pbar:
                    for batch in dataloader:
                        new_uids.extend(batch["uid"])

                        clip_text_features.append(
                            clip_model.encode_text(
                                clip_tokenizer(batch["text"]).to(DEFAULT_DEVICE)
                            ).cpu()
                        )

                        pbar.update(len(batch["uid"]))

            clip_text_features = (
                torch.cat(clip_text_features, dim=0).numpy().astype("float16")
            )
            uid_to_text_feature = {
                uid: text_feature
                for uid, text_feature in zip(new_uids, clip_text_features)
            }
            reordered_text_features = [
                uid_to_text_feature[uid] for uid in clip_features_dict["uids"]
            ]

            clip_text_features = np.stack(reordered_text_features, axis=0)
            clip_features_dict["text_features"] = clip_text_features
            compress_pickle.dump(clip_features_dict, clip_features_path)

            return clip_features_dict

    def retrieve_with_name_and_desc(
        self,
        object_names: Optional[Sequence[str]],
        object_descriptions: Optional[Sequence[str]],
        threshold=28,
    ) -> Sequence[Tuple[str, float]]:
        assert (
            object_names is not None or object_descriptions is not None
        ), "At least one of object_names or object_descriptions must be provided"

        if object_names is not None and object_descriptions is not None:
            assert len(object_names) == len(
                object_descriptions
            ), f"object_names and object_descriptions must have the same length but got {len(object_names)} and {len(object_descriptions)}"

        if object_names is None:
            object_names = [None] * len(object_descriptions)

        if object_descriptions is None:
            object_descriptions = [None] * len(object_names)

        img_queries = []
        text_queries = []
        for object_name, object_desc in zip(object_names, object_descriptions):
            q_img_list = []
            q_text_list = []

            if object_name is not None:
                q_img_list.append(f"a 3D model of {object_name}")
                q_text_list.append(f"a {object_name}")

            if object_desc is not None:
                if object_name is None:
                    q_img_list.append(f"a 3D model of {object_desc}")
                else:
                    q_img_list.append(object_desc)
                q_text_list.append(object_desc)

            img_queries.append(", ".join(q_img_list))
            text_queries.append(", ".join(q_text_list))

        return self.retrieve(
            img_queries=img_queries, text_queries=text_queries, threshold=threshold
        )

    def retrieve(
        self,
        img_queries: Sequence[str],
        text_queries: Optional[Sequence[str]] = None,
        threshold=28,
    ) -> Sequence[Tuple[str, float]]:
        if text_queries is None:
            text_queries = img_queries

        with torch.no_grad():
            img_query_features_clip = self.clip_model.encode_text(
                self.clip_tokenizer(img_queries)
            )
            img_query_features_clip = F.normalize(img_query_features_clip, p=2, dim=-1)

            text_query_features_clip = self.clip_model.encode_text(
                self.clip_tokenizer(text_queries)
            )
            text_query_features_clip = F.normalize(
                text_query_features_clip, p=2, dim=-1
            )

        clip_img_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", img_query_features_clip, self.clip_img_features
        )
        clip_img_similarities = torch.max(clip_img_similarities, dim=-1).values

        clip_text_similarities = 100 * torch.einsum(
            "ij, lj -> il", text_query_features_clip, self.clip_text_features
        )

        joint_similiarities = clip_img_similarities + clip_text_similarities

        threshold_indices = torch.where(clip_img_similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = joint_similiarities[query_index, asset_index].item()

            asset_id = self.asset_ids[asset_index]

            similarity = self.database[asset_id].get("blender_thor_similarity", 1.0)
            if similarity >= self.blender_thor_similarity_threshold:
                unsorted_results.append((asset_id, score))

        # Sorting the results in descending order by score
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        return results

    def compute_size_difference(self, target_size, candidates):
        candidate_sizes = []
        for uid, _ in candidates:
            size_list = (100 * get_bbox_dims_vec(self.database[uid])).tolist()
            size_list.sort()
            candidate_sizes.append(size_list)

        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 10)
            )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference
