# zbggGraphNet3: MeshCrafter Graph2Seq

The main code base of **MeshCrafter**, a research project on learning finite-element surface-mesh refinement. The project was funded by TÜBİTAK 2209-B.

The idea:
- Treat a refined shell mesh (Nastran `CTRIA3`/`CQUAD4`) as a **sequence of local patch tokens**.
- Train a graph-to-sequence model that reads the coarse ("raw") mesh as a graph and writes the token sequence of the refined mesh.

```
raw .nas ──► PyG graph ──► GAT encoder ──► context ──► Transformer decoder ──► patch tokens ──► reassembled .nas
                                                             ▲
refined .nas ──► 1-ring patches ──► WL-hash ──► vocabulary ──┘ (training targets)
```

> **Status:** research prototype. The pipeline runs end to end: vocabulary → training pairs → training → generation → `.nas` export. Reassembling patches into one clean mesh is still the open problem; `generate.py` → `generate_2.py` → `generate_3.py` are successive attempts at it. See [Known limitations](#known-limitations).

## How it works

### Phase 1: patch vocabulary (`faz_1_pipeline.py`)

1. Parse each `.nas` file. Nodes come from `GRID` cards; elements come from `CTRIA3` and `CQUAD4` cards.
2. For every node, extract its **1-ring patch** (`utils/patch_extractor.py`):
   - the centre node and its neighbours;
   - coordinates centred on the centre node;
   - local element connectivity.
3. Hash each patch's topology with the **Weisfeiler–Lehman graph hash** (networkx). Patches with the same hash are treated as the same "word".
4. Sort the hashes by frequency and keep one template patch per hash.
5. Write `data_vocabulary/ai_vocabulary.json`.

Special tokens: `0 <PAD>`, `1 <START>`, `2 <END>`, `3 <UNK>`. The patch tokens start at 4.

### Phase 2: training pairs (`faz_2_pipeline.py`)

For each matched raw/refined file pair, Phase 2 builds an input graph and a target sequence.

**Input (raw mesh):** a `torch_geometric.data.Data` graph with:
- `x`: 4 node features: valence, mean aspect ratio, mean skewness and mean scaled Jacobian of the adjacent elements;
- bidirectional `edge_index`;
- `pos`.

**Target (refined mesh):**
1. Put the nodes and elements in **canonical order** (`utils/canonical_order.py`). Nodes are sorted Z→Y→X. Elements are traversed by BFS over adjacency, so neighbouring patches stay neighbours in the sequence.
2. Map each node's patch hash to a token ID.
3. The sequence is `[START] + ids + [END]`.

Each pair is saved as `<stem>.pt`.

### Model (`models/`)

| Part | Implementation | Defaults |
|---|---|---|
| `HourGlassGNNEncoder` | Linear projection → `GATConv` × L (BatchNorm, ReLU, skip connections) → global pooling → MLP | hidden 256, 4 layers, 4 heads |
| `AutoregressiveMeshDecoder` | Token embedding + sinusoidal positions → `nn.TransformerDecoder` (causal self-attention, cross-attention to the encoder context) → linear head | 4 layers, 8 heads, FFN 1024, max length 2048 |

The vocabulary has 33 entries. The output layer is padded to 64 for memory alignment.

### Training (`training/train.py`)

- Uses teacher forcing with cross-entropy, ignoring `<PAD>`.
- Optimizer: AdamW (lr 3e-4, weight decay 1e-4) with a cosine LR schedule and gradient clipping at 1.0.
- Batch size 4, 50 epochs, 85/15 train/validation split.
- The best checkpoint is written to `checkpoints/best_model.pth`.

All settings are in the `Config` class at the top of the file.

### Generation (`inference/`)

1. Encode the raw graph.
2. Decode greedily, with a repetition penalty and a temperature setting.
3. Place each predicted patch template back in 3D.
4. Export the result as Nastran free-field `.nas`.
5. Optionally, show raw, refined and predicted meshes side by side in PyVista.

Each version changes how the patches are placed and joined:

| Script | Placement / merging |
|---|---|
| `generate.py` | Patch is translated to its raw node and denormalized |
| `generate_2.py` | Adds rotation to the local surface normal (Rodrigues), KD-tree vertex welding (Union-Find) and Laplacian smoothing |
| `generate_3.py` | Adds per-patch scaling so neighbouring patches touch, and a weld tolerance that adapts to the patch radius |

## Repository layout

```
faz_1_pipeline.py          Phase 1 – build the patch vocabulary
faz_2_pipeline.py          Phase 2 – build (graph, token sequence) .pt pairs
models/encoder.py          GAT graph encoder
models/decoder.py          autoregressive Transformer decoder
training/train.py          dataset, Graph2Seq wrapper, trainer
inference/generate*.py     generation + mesh reassembly (v1–v3)
utils/
  parser_utils.py          Nastran GRID / CTRIA3 / CQUAD4 parser
  patch_extractor.py       1-ring patch extraction + WL hash
  canonical_order.py       node / element canonical ordering
  normalizasyon.py         per-mesh normalization (parameters stored in CSV)
  codebook_builder.py      hash frequency analysis, Pareto plot, codebook
  build_codebook.py, export_ai_vocabulary.py   vocabulary export helpers
scripts/
  analyze_topology_codebook.py   inspect representative patches per hash
  compare_meshes.py              raw vs refined statistics
  visualize_mesh.py              side-by-side PyVista viewer
```

## Data layout

The data is not part of the repository. The scripts expect a `DATA/` folder **next to** the repository:

```
<workspace>/
├── DATA/
│   ├── RAW_NAS/ , REFINED_NAS/     paired meshes with matching file names
│   ├── RAW_NORMALIZED/ , REFINED_NORMALIZED/   output of utils/normalizasyon.py
│   ├── PT_FILES/                   Phase 2 output
│   └── normalize_parameter.csv
└── zbggGraphNet3/                  this repository
```

Folder defaults come from a `config.yaml` in the repository root (not committed). The pipeline CLIs accept explicit paths that override it:

```yaml
raw_dir: ../DATA/RAW_NAS
refined_dir: ../DATA/REFINED_NAS
raw_output_dir: ../DATA/RAW_NORMALIZED
refined_output_dir: ../DATA/REFINED_NORMALIZED
normalized_parameter_csv_dir: ../DATA
torch_path_dir: ../DATA/PT_FILES
logs_dir: Log
```

## Usage

```bash
pip install torch torch_geometric networkx numpy scipy trimesh pyvista pyyaml

python utils/normalizasyon.py          # normalize raw + refined meshes (paths from config.yaml)
python faz_1_pipeline.py --source refined
python faz_2_pipeline.py --raw-dir ../DATA/RAW_NORMALIZED --refined-dir ../DATA/REFINED_NORMALIZED \
                         --vocab data_vocabulary/ai_vocabulary.json --output-dir ../DATA/PT_FILES
python training/train.py
python inference/generate_3.py --pt-file ../DATA/PT_FILES/<mesh>.pt --output out.nas
```

Run any script with `--help` to see all of its options.

## Known limitations

- **Small vocabulary.** WL hashing captures topology only, and 1-ring patches of quad-dominant meshes collapse into a few dozen classes. Each token therefore carries no geometry; geometry comes from the template plus the raw node position.
- **The stitch loss has no effect yet.** `calculate_stitch_loss` is computed on detached logits, so training is effectively cross-entropy only.
- **Reassembly is heuristic.** Overlapping patches are joined by welding and smoothing, not by a learned or constrained method. Gaps and duplicate elements can remain.
- **Single context vector.** The encoder pools the whole raw graph into one vector, which limits how much detail the decoder can recover on large meshes.

## Related repositories

- **NasMeshLib**: turns `.nas` meshes into graph features (an earlier preprocessing stage).
- **Ansa_Macro**: ANSA batch script that generates the paired raw/refined meshes.
- **zbggGraphNet**, **zbggGraphNet2**, **RefineGraphnet**, **generative_ai**: earlier iterations of this project.
