"""config.py - every setting of the HNSW experiments, in one place."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- data
# Task 1 output (Hugging Face pbcong/sc4020), CLIP ViT-B/32, 512-d, L2-normalised.
#   coco : MS-COCO 2014 train+val, 122,287 corpus images + 1,000 held-out query images and their
#          first captions (text query i describes image query i -> paired).
#   laion: VIBE laion-clip-512-normalized, ~999k corpus images + 1,000 held-out query images and the
#          1,000 VIBE text queries (text query i is unrelated to image query i -> unpaired).
SOURCES = {
    "coco": "https://huggingface.co/datasets/pbcong/sc4020/resolve/main/coco/coco_prepared.hdf5",
    "laion": "https://huggingface.co/datasets/pbcong/sc4020/resolve/main/laion/laion_prepared.hdf5",
}
PAIRED = {"coco": True, "laion": False}   # paired -> SE of Δ recall uses per-query differences
DATA_DIR = os.path.join(HERE, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DATASETS = {   # name -> (source, number of corpus vectors (None = all), how the subset is taken)
    "coco_full": ("coco", None, None),              # 122,287 vectors - official COCO results
    "coco_5k": ("coco", 5000, "random"),            # spec-sized; used for the from-scratch implementation
    "laion_full": ("laion", None, None),            # 999,448 vectors (~35 min build, ~2.5 GB RAM)
    "laion_100k": ("laion", 100000, "first"),       # first 100,000 corpus rows (rows 0..99,999)
    "laion_5k": ("laion", 5000, "random"),          # from-scratch implementation
}
# "random": np.random.default_rng(SUBSET_SEED).choice(n_corpus, n, replace=False), sorted
# "first" : the first n rows of the corpus, in file order


def hdf5_path(source):
    return os.path.join(RAW_DATA_DIR, f"{source}_prepared.hdf5")


def source_of(dataset):
    return DATASETS[dataset][0]


SUBSET_SEED = 0
GT_K = 100                 # depth of the exact ground truth

# ---------------------------------------------------------------- HNSW (fixed by the project spec)
M_GRAPH = 32               # FAISS: links per node on layers >= 1, 2*M = 64 on layer 0
EF_CONSTRUCTION = 200      # beam width while building; index is built once
EF_SWEEP = [16, 32, 64, 128, 256]   # beam width while searching

# ---------------------------------------------------------------- evaluation
K = 10                     # results per query for recall@1/@10 and timing
THREADS = 1                # single thread for building and searching
TIMING_REPEATS = {"faiss": 5, "basic": 1, "exact": 1}   # passes per query; per-query time = median
TIMING_WARMUP = 200

# ---------------------------------------------------------------- outputs
RESULTS_DIR = os.path.join(HERE, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")          # per-run files: <dataset>/<method>*.csv/json
ALL_RESULTS = os.path.join(RESULTS_DIR, "hnsw_all_results.csv")
