import json
import torch
import pickle

class ObjaverseRetriever():
    def __init__(self, clip_model, clip_preprocess, clip_tokenizer, sbert_model, version, retrieval_threshold):
        self.database = json.load(open(f"./data/objaverse_holodeck/{version}/objaverse_holodeck_database.json", "r"))
        self.asset_ids = list(self.database.keys())

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model

        self.clip_features = pickle.load(open(f"data/objaverse_holodeck/{version}/objaverse_holodeck_features_clip_3.p", "rb")).float() # clip features
        self.sbert_features = pickle.load(open(f"data/objaverse_holodeck/{version}/objaverse_holodeck_description_features_sbert.p", "rb")).float() # sbert features
        self.retrieval_threshold = retrieval_threshold

        self.use_text = True


    def retrieve(self, queries, threshold=28):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(self.clip_tokenizer(queries))
            query_feature_clip /= query_feature_clip.norm(dim=-1, keepdim=True)
        
        clip_similarities = query_feature_clip @ self.clip_features.T * 100
        clip_similarities = clip_similarities.reshape((len(queries), len(self.asset_ids), 3))
        clip_similarities = torch.max(clip_similarities, dim=2).values

        query_feature_sbert = self.sbert_model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        sbert_similarities = query_feature_sbert @ self.sbert_features.T

        if self.use_text: similarities = clip_similarities + sbert_similarities
        else: similarities = clip_similarities

        threshold_indices = torch.where(clip_similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            unsorted_results.append((self.asset_ids[asset_index], score))

        # Sorting the results in descending order by score
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        return results
    

    def compute_size_difference(self, target_size, candidates):
        candidate_sizes = []
        for uid, _ in candidates:
            size = self.database[uid]['assetMetadata']['boundingBox']
            size_list = [size['x'] * 100, size['y'] * 100, size['z'] * 100]
            size_list.sort()
            candidate_sizes.append(size_list)
        
        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1)/100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append((uid, score - size_difference[i] * 10))

        # sort the candidates by score
        candidates_with_size_difference = sorted(candidates_with_size_difference, key=lambda x: x[1], reverse=True)

        return candidates_with_size_difference