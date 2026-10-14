import argparse
import json
import os
import sys

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_ROOT, N_QUERIES, SEED, exact_topk, l2_normalize, write_prepared  # noqa: E402

COCO_ROOT = os.path.expanduser(os.environ.get("COCO_ROOT", "~/coco2014"))
MODEL, PRETRAINED = "ViT-B-32-quickgelu", "openai"


class Images(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]).convert("RGB"))


def load_coco():
    images, captions = {}, {}
    for split in ["train2014", "val2014"]:
        with open(os.path.join(COCO_ROOT, "annotations", f"captions_{split}.json")) as f:
            data = json.load(f)
        for img in data["images"]:
            images[img["id"]] = os.path.join(split, img["file_name"])
        for ann in data["annotations"]:
            captions.setdefault(ann["image_id"], []).append((ann["id"], ann["caption"].strip()))
    image_ids = np.array(sorted(images), dtype=np.int64)
    for caps in captions.values():
        caps.sort()
    return image_ids, images, captions


@torch.no_grad()
def encode_images(model, preprocess, paths, device, batch_size):
    loader = DataLoader(Images(paths, preprocess), batch_size=batch_size, num_workers=32)
    out = []
    for i, batch in enumerate(loader):
        out.append(model.encode_image(batch.to(device)).float().cpu())
        if i % 20 == 0:
            print(f"  images {i * batch_size}/{len(paths)}", flush=True)
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size):
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(model.encode_text(tokenizer(texts[i:i + batch_size]).to(device)).float().cpu())
        if (i // batch_size) % 50 == 0:
            print(f"  texts {i}/{len(texts)}", flush=True)
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    image_ids, image_files, captions = load_coco()
    print(f"{len(image_ids)} images, {sum(len(c) for c in captions.values())} captions")

    first_captions = [captions[i][0][1] for i in image_ids]
    img_cache = os.path.join(DATA_ROOT, "coco", "clip_vitb32_images.npy")
    cap_cache = os.path.join(DATA_ROOT, "coco", "clip_vitb32_first_captions.npy")
    if not (os.path.exists(img_cache) and os.path.exists(cap_cache)):
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=args.device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(MODEL)
    if not os.path.exists(img_cache):
        paths = [os.path.join(COCO_ROOT, image_files[i]) for i in image_ids]
        np.save(img_cache, encode_images(model, preprocess, paths, args.device, args.batch_size))
    if not os.path.exists(cap_cache):
        np.save(cap_cache, encode_texts(model, tokenizer, first_captions, args.device, args.batch_size))
    img_vecs = l2_normalize(np.load(img_cache))
    cap_vecs = l2_normalize(np.load(cap_cache))

    rng = np.random.default_rng(SEED)
    q_rows = np.sort(rng.choice(len(image_ids), N_QUERIES, replace=False))
    is_query = np.zeros(len(image_ids), dtype=bool)
    is_query[q_rows] = True
    corpus_rows = np.flatnonzero(~is_query)
    query_image_ids = image_ids[q_rows]

    corpus = img_vecs[corpus_rows]
    image_q = img_vecs[q_rows]
    text_q = cap_vecs[q_rows]
    gt_text_dist, gt_text = exact_topk(corpus, text_q)
    gt_image_dist, gt_image = exact_topk(corpus, image_q)

    write_prepared("coco", {
        "corpus": corpus,
        "queries_text": text_q,
        "queries_image": image_q,
        "gt_text": gt_text,
        "gt_image": gt_image,
        "gt_text_dist": gt_text_dist,
        "gt_image_dist": gt_image_dist,
        "learn_text": cap_vecs[corpus_rows],
        "corpus_coco_ids": image_ids[corpus_rows],
        "query_coco_ids": query_image_ids,
    }, {
        "source": f"MS-COCO 2014 train+val, open_clip {MODEL}/{PRETRAINED}",
        "metric": "inner_product",
        "seed": SEED,
    })

    meta = {
        "corpus_files": [image_files[i] for i in image_ids[corpus_rows]],
        "query_files": [image_files[i] for i in query_image_ids],
        "query_captions": [first_captions[r] for r in q_rows],
    }
    with open(os.path.join(DATA_ROOT, "coco", "coco_meta.json"), "w") as f:
        json.dump(meta, f)
    print("wrote coco_meta.json")


if __name__ == "__main__":
    main()
