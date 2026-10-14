// Builds the technical-review report. Usage (from report/): npm install && node build.js [out.docx]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, LevelFormat, Footer, Header, PageNumber, TableLayoutType, TabStopType,
} = require("docx");
const { math } = require("./omml");

const FIG = path.join(__dirname, "..", "results", "figures") + path.sep;
const OUT = process.argv[2] || path.join(__dirname, "group_XX_report.docx");
const TEXT_W = 9026; // A4 width 11906 minus 2 x 1440 margins (DXA)
const FONT = "Times New Roman";
const R = String.raw;

// ---------- inline markup: $math$, **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("$")) out.push(math(t.slice(1, -1)));
    else if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const BODY = { after: 120, line: 264 };
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: BODY, alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (n, t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(n ? `${n} ${t}` : t)] });
const H2 = (n, t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(`${n} ${t}`)] });
const H3 = (t) => new Paragraph({ spacing: { before: 140, after: 60 }, keepNext: true, children: [new TextRun({ text: t, bold: true, italics: true })] });
const B = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(text),
  spacing: { after: 70, line: 256 }, alignment: AlignmentType.JUSTIFIED });
const N = (text, ref = "nums") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text),
  spacing: { after: 70, line: 256 }, alignment: AlignmentType.JUSTIFIED });

// numbered display equation
let eqCount = 0;
const EN = {};
function EQ(src, key) {
  eqCount += 1;
  if (key) EN[key] = eqCount;
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 80, after: 120 },
    children: [new TextRun("\t"), math(src), new TextRun(`\t(${eqCount})`)],
  });
}

function figure(file, w, h, num, caption, widthIn = 6.1) {
  const wpx = Math.round(widthIn * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + file),
        transformation: { width: wpx, height: Math.round(wpx * h / w) },
        altText: { title: `Figure ${num}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 220, line: 240 }, indent: { left: 360, right: 360 },
      children: [new TextRun({ text: `Figure ${num}. `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
  ];
}

// three-line (booktabs-style) table
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const THICK = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
const THIN = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
function table(num, caption, header, rows, weights, note, align) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / total) * TEXT_W));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, kind) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { left: NONE, right: NONE, top: kind === "head" ? THICK : NONE,
      bottom: kind === "head" ? THIN : kind === "last" ? THICK : NONE },
    margins: { top: 45, bottom: 45, left: 80, right: 80 },
    children: [new Paragraph({ alignment: (align ? align[i] : i === 0) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(text), kind === "head" ? { size: 18, bold: true } : { size: 18 }) })],
  });
  return [
    new Paragraph({ keepNext: true, spacing: { before: 200, after: 80, line: 240 }, alignment: AlignmentType.JUSTIFIED,
      children: [new TextRun({ text: `Table ${num}. `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
      borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, "head")) }),
        ...rows.map((r, j) => new TableRow({ cantSplit: true,
          children: r.map((c, i) => cell(c, i, j === rows.length - 1 ? "last" : "body")) }))] }),
    new Paragraph({ spacing: { before: 60, after: 200, line: 240 }, children: note ? runs(note, { size: 17, italics: true }) : [] }),
  ];
}

// algorithm box: title line between rules, numbered steps, closing rule
function algorithm(title, lines) {
  const ruled = (top, bottom) => ({ top: top ? THICK : NONE, bottom: bottom ? (bottom === "thin" ? THIN : THICK) : NONE, left: NONE, right: NONE });
  const row = (children, b) => new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: TEXT_W, type: WidthType.DXA },
    borders: b, margins: { top: 30, bottom: 30, left: 80, right: 80 }, children })] });
  let step = 0;
  const body = lines.map(([indent, text]) => {
    const io = /^\*\*(Input|Output)/.test(text);
    if (!io) step += 1;
    return new Paragraph({ spacing: { after: 30, line: 240 }, indent: { left: 420 * indent + 360, hanging: 360 },
      children: [new TextRun({ text: io ? "" : `${step}:  `, size: 19 }), ...runs(text, { size: 19 })] });
  });
  return [
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W], layout: TableLayoutType.FIXED,
      borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
      rows: [row([new Paragraph({ children: runs(title, { size: 19 }) })], ruled(true, "thin")), row(body, ruled(false, true))] }),
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}

// ================================================================== CONTENT
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// ---- title block
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: "When the Query Speaks a Different Language:", bold: true, size: 34 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Hashing, Quantization and Graph Indexes for Cross-Modal Nearest Neighbour Search", bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Group XX: [member names]", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 320 },
    children: [new TextRun({ text: "SC4020 Project 1 — Technical Review (Similarity Search), Nanyang Technological University", size: 20, italics: true })] }),
);

// ---- abstract
const ABS = { left: 567, right: 567 };
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Abstract", bold: true, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: ABS, spacing: { after: 100, line: 252 }, children: runs(
    "Text-to-image search engines embed captions and photos into a shared space with a contrastive model such as CLIP, then answer each caption with an approximate nearest neighbour (ANN) index built over the image vectors. Such indexes are normally built and tuned with the stored data, yet the queries they serve come from the text encoder, which CLIP places in a different region of the space (the *modality gap*). We present a controlled comparison of three ANN families — random-projection LSH (hashing), Product Quantization (quantization) and HNSW (graph) — on MS-COCO (122k images) and LAION (1M images), searching every index with both image and text queries under identical settings. All three methods lose recall on text queries at every operating point, and, contrary to the intuition that a data-independent method should be unaffected, LSH loses the most (up to 0.40 recall@100), ahead of PQ (0.17) and HNSW (0.19). We explain these results with a score-margin model: text queries have *low neighbour contrast*, which magnifies the error of approximate scoring (LSH, PQ), and they lie *far from all images*, which weakens greedy graph routing (HNSW), increasingly so as the corpus grows. A binomial analysis of LSH predicts the measured loss almost exactly. Guided by the model, we propose three query-aware construction methods that preserve the exact ground truth: caption-guided HNSW edges raise COCO text recall from 0.88 to 0.95 at no memory cost, query-weighted OPQ raises PQ text recall from 0.51 to 0.79 at 64 bytes per vector, and modality-aware centring roughly doubles LSH text recall at mid-range code lengths.", { size: 20 }) }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: ABS, spacing: { after: 280 }, children: [
    new TextRun({ text: "Keywords: ", bold: true, size: 20 }),
    new TextRun({ text: "approximate nearest neighbour search, locality-sensitive hashing, product quantization, HNSW, cross-modal retrieval, CLIP, out-of-distribution queries", italics: true, size: 20 })] }),
);

// ---- 1 introduction
add(H1("1", "Introduction"),
  P("Similarity search — retrieving the $k$ database items most similar to a query — underpins image search, recommendation and retrieval-augmented generation. The exact solution, a linear scan, compares the query with every stored vector and therefore costs $O(nd)$ for $n$ vectors of dimension $d$; on our one-million-image LAION corpus a single-threaded scan takes 104 ms per query. Approximate nearest neighbour (ANN) indexes trade a small loss in accuracy for large gains in speed or memory, either by making each comparison cheaper (compact codes, as in LSH and PQ) or by making fewer comparisons (partitioning or graph navigation)."),
  P("Contrastive vision–language models such as CLIP (Radford et al., 2021) make *cross-modal* search possible with the same machinery: a caption and a photo are mapped into one 512-dimensional space, and the photos closest to the caption are returned. The two modalities, however, are not mixed in that space. Each occupies its own narrow cone, and the cones are far apart (Liang et al., 2022): in our data a caption and its own photo have cosine similarity of only about 0.3. An index whose structure was learned from image vectors is therefore queried from a region it has never seen. The VIBE benchmark (Jääsaari et al., 2025) identifies such out-of-distribution (OOD) queries as a principal weakness of current ANN methods, but it does not explain *why* different index families fail, nor how much of the failure is intrinsic to the queries themselves."),
  P("This report studies that question through a controlled experiment in which the index, its parameters and the ground truth are held fixed and only the query modality changes. We select one representative of each major ANN family, so that differences can be attributed to mechanisms rather than implementations: random-projection **LSH** (hashing; data-independent) and **Product Quantization** (quantization; codebooks learned from data), both covered in the course, and **HNSW** (graph; neighbourhood structure learned from data), which lies outside the syllabus and is the default index of pgvector, Milvus and Weaviate. We address three research questions:"),
  N("**RQ1.** How do hashing, quantization and graph indexes trade accuracy, speed and memory on CLIP embeddings?", "rq"),
  N("**RQ2.** How much recall does each family lose when the query is a caption rather than an image, and can a larger budget recover it?", "rq"),
  N("**RQ3.** Which property of text queries causes the loss, and can index construction be adapted to it without changing the ground truth?", "rq"),
  P("Our contributions are as follows:"),
  B("A controlled cross-modal benchmark of three ANN families on two datasets that differ in scale (122k vs 1M images) and text style (clean captions vs web text)."),
  B("A *score-margin* account of the modality penalty, which identifies two query properties — low neighbour contrast and distance from the data — and predicts which family each property affects. For LSH the account is quantitative: a binomial model reproduces the measured swap rates and the factor-of-eight increase in code length that text queries require."),
  B("Three query-aware construction methods, one per family, that use a sample of training captions but leave the search algorithm and the exact ground truth unchanged, together with ablations of their components and two instructive negative results."),
  P("Section 2 reviews the search problem and the modality gap. Section 3 describes the three methods with the formulation used in the course, then develops the score-margin model and derives the query-aware extensions from it. Section 4 reports the experiments, Section 5 discusses strengths, weaknesses and limitations, and Section 6 concludes."),
);

// ---- 2 background
add(H1("2", "Background and Related Work"),
  H2("2.1", "Nearest neighbour search"),
  P("Let $X=\\c{x_{1},…,x_{n}}⊂ℝ^{d}$ be the database and $q∈ℝ^{d}$ a query. Given a similarity $s(q,x)$, the exact $k$-nearest-neighbour set is"),
  EQ(R`N_{k}(q)=\lim{argmax}{S⊂X, \abs{S}=k} \sum_{x∈S}{s(q,x)}`, "knn"),
  P("All vectors in this study are L2-normalised, so the inner product, the cosine similarity and the Euclidean distance induce the same ranking:"),
  EQ(R`\n{q−x}^{2}=\n{q}^{2}+\n{x}^{2}−2\a{q,x}=2−2\a{q,x}=2−2 cos θ(q,x)`, "equiv"),
  P(R`where $θ(q,x)$ is the angle between the vectors. We therefore use $s(q,x)=\a{q,x}$ throughout. An ANN method returns an approximate set $\tilde{N}_{k}(q)$, and its quality is measured by recall against the exact answer,`),
  EQ(R`Recall@k=\frac{1}{\abs{Q}}\sum_{q∈Q}{\frac{\abs{\tilde{N}_{k}(q)∩N_{k}(q)}}{k}}`, "recall"),
  P("where $Q$ is the query set. Recall measures fidelity to exact search, not human relevance; a separate retrieval metric would mix the index's error with the embedding model's error."),

  H2("2.2", "Contrastive embeddings and the modality gap"),
  P(R`CLIP trains an image encoder $f_{I}$ and a text encoder $f_{T}$ on image–caption pairs with a symmetric InfoNCE loss. For a batch of $B$ pairs $(i_{j},t_{j})$ and temperature $τ$, the image-to-text term is`),
  EQ(R`L_{I→T}=−\frac{1}{B}\sum_{j=1}^{B}{log\frac{exp(\a{f_{I}(i_{j}),f_{T}(t_{j})}/τ)}{\sum_{l=1}^{B}{exp(\a{f_{I}(i_{j}),f_{T}(t_{l})}/τ)}}}`, "clip"),
  P(R`The loss only asks a matched pair to be *more* similar than mismatched pairs; it does not ask the two modalities to overlap. Liang et al. (2022) show that, as a result, each encoder maps its inputs into a narrow cone and the two cones are separated by a roughly constant offset. Writing $μ_{I}=E[f_{I}(i)]$ and $μ_{T}=E[f_{T}(t)]$ for the modality means, we quantify the gap by $cos(μ_{I},μ_{T})$, which is 0.31 on COCO and 0.36 on LAION (Section 4.2). A text query is thus, from the point of view of an image index, an out-of-distribution query.`),

  H2("2.3", "Related work"),
  P("**ANN benchmarks.** ann-benchmarks (Aumüller et al., 2020) and the FAISS library (Douze et al., 2024) established the recall–throughput methodology we follow. VIBE (Jääsaari et al., 2025) adds modern embedding datasets, including cross-modal LAION queries, and reports that graph indexes lose accuracy on them; we adopt its protocol and its LAION data, and extend the analysis to hashing and quantization and to the mechanism behind the loss."),
  P("**OOD-aware graph indexes.** RoarGraph (Chen et al., 2024) and OOD-DiskANN (Jaiswal et al., 2022) use a sample of training queries to add or reshape graph edges so that routing works for queries outside the data distribution. Our caption-guided edges (Section 3.8) follow this idea in its simplest form, inserting edges into an existing HNSW graph without rebuilding it."),
  P("**Query-aware quantization.** OPQ (Ge et al., 2014) learns a rotation that balances variance across PQ sub-spaces. ScaNN (Guo et al., 2020) observes that, for inner-product search, the quantization error parallel to the query matters more than the orthogonal error, and weights the loss accordingly. Our query-weighted PQ (Section 3.7) is a closed-form variant that weights the error by the second-moment matrix of a query distribution."),
);

// ---- 3 methods
add(H1("3", "Methods"),
  P("This section first describes the three methods and the exact baseline (Sections 3.1–3.4), then introduces a common model of their failure on out-of-distribution queries (Section 3.5) and the query-aware extensions derived from it (Sections 3.6–3.8). Table 1 summarises their storage and query cost; the parameters we sweep are listed in the last column."),
  ...table(1, "The compared methods. $n$: database size, $d$: dimension (512), $b$: LSH code length in bits, $M$: number of PQ sub-spaces or HNSW links per node, $K$: centroids per sub-space (256), $ef$: HNSW search beam width.",
    ["Method", "Family", "Learns from data", "Bytes per vector", "Query cost", "Parameter swept"],
    [["LSH (Charikar, 2002)", "Hashing", "No", "b / 8", "O(nb) bit ops", "b = 64 … 4,096"],
     ["PQ (Jégou et al., 2011)", "Quantization", "Yes (k-means)", "M", "O(Kd + nM)", "M = 8, 16, 32, 64"],
     ["HNSW (Malkov & Yashunin, 2020)", "Graph", "Yes (graph)", "4d + 8M", "≈ O(ef·M·d·log n)", "ef = 100 … 1,024"],
     ["Exact (linear scan)", "—", "—", "4d", "O(nd)", "—"]],
    [2.6, 1.3, 1.4, 1.3, 1.9, 1.9]),

  H2("3.1", "Locality-sensitive hashing by random projection"),
  P(R`Following the course formulation, we draw $b$ random vectors $w_{1},…,w_{b}$ from a standard Gaussian and define one binary hash function per vector ($1[·]$ is 1 if the condition holds and 0 otherwise),`),
  EQ(R`h_{i}(x)=1\b{w_{i}^{⊤}x>0}, H(x)=(h_{1}(x),…,h_{b}(x))∈\c{0,1}^{b}`, "hash"),
  P(R`Geometrically, $w_{i}$ is the normal of a hyperplane through the origin and $h_{i}$ records on which side of it a vector lies. Two vectors are compared by the Hamming distance of their codes, $D_{H}(q,x)=popcount(H(q) ⊕ H(x))$, which a CPU computes with one XOR and one population count per 64 bits. Charikar (2002) showed that a random hyperplane separates two vectors with probability proportional to their angle:`),
  EQ(R`Pr\b{h_{i}(q)≠h_{i}(x)}=\frac{θ(q,x)}{π} ⇒ E\b{D_{H}(q,x)}=\frac{b θ(q,x)}{π}`, "charikar"),
  P(R`so the Hamming distance is an unbiased estimate of the angle, and by Eq. (${EN.equiv}) of the inner product. Since the bits are independent Bernoulli trials, $D_{H}$ is binomial and the estimate becomes sharper as $b$ grows; the lecture's observation that more hash functions give more accurate top-$k$ results is the law of large numbers applied to Eq. (${EN.charikar}).`),
  P(R`The course presents two ways to search with LSH codes: a linear scan over the codes, or hash tables whose buckets are probed in Hamming order. We use the linear scan (FAISS IndexLSH), which ranks every database code by $D_{H}$ and is the standard way to obtain a full top-100 list for recall measurement. A code costs $b/8$ bytes, so 512 bits (64 bytes) compress a 2,048-byte float vector 32×. We generate the hyperplanes as the rows of a seeded random rotation and average every LSH result over three seeds; the standard deviation of recall across seeds is below 0.01.`),
  P(R`**Resolution of the estimator.** Recall depends on whether a true neighbour $u$ (say, the 100th) stays ahead of a non-neighbour $v$ with $θ(q,u)<θ(q,v)$. For each hyperplane let $D_{i}=1[h_{i}(q)≠h_{i}(v)]−1[h_{i}(q)≠h_{i}(u)]$. Its mean is the angular margin $δ=(θ(q,v)−θ(q,u))/π$, and since $D_{i}≠0$ exactly when the hyperplane separates $u$ from $v$, its variance is $θ(u,v)/π−δ^{2}$. Summing over $b$ independent hyperplanes, the codes rank $v$ above $u$ with probability`),
  EQ(R`Pr[swap]≈Φ(−z), z=δ\sqrt{\frac{b}{θ(u,v)/π−δ^{2}}}`, "swap"),
  P(R`where $Φ$ is the standard normal CDF. The key consequence is that $z$ grows only with $δ\sqrt{b}$: **halving the angular margin requires four times as many bits** to keep the same swap probability. This is independent of where the data lie, so we initially expected LSH to be insensitive to the query modality. Section 3.5 explains why the margin itself depends on the query, and Section 4.5 tests Eq. (${EN.swap}) against measurement.`),

  H2("3.2", "Product quantization"),
  P(R`PQ (Jégou et al., 2011) splits each vector into $M$ contiguous sub-vectors $x=(x^{(1)},…,x^{(M)})$ with $x^{(m)}∈ℝ^{d/M}$, and in each sub-space $m$ runs k-means on the database to obtain a codebook $C_{m}=\c{c_{m,1},…,c_{m,K}}$ of $K=256$ centroids:`),
  EQ(R`\lim{min}{C_{m}} \sum_{x∈X}{\n{x^{(m)}−c_{m,i_{m}(x)}}^{2}}, i_{m}(x)=\lim{argmin}{j} \n{x^{(m)}−c_{m,j}}`, "kmeans"),
  P(R`A vector is stored as its $M$ centroid indices $(i_{1}(x),…,i_{M}(x))$, one byte each, and is reconstructed as $\hat{x}=(c_{1,i_{1}(x)},…,c_{M,i_{M}(x)})$. With $M=64$, a 2,048-byte vector becomes a 64-byte code (32× smaller); with $M=8$, 256× smaller. The product structure is what makes this tractable: the $M$ codebooks jointly represent $K^{M}=2^{8M}$ reconstruction points while storing only $M·K$ centroids.`),
  P(R`At query time we use the *asymmetric* distance computation (ADC) from the course: only database vectors are quantized, the query is kept exact. For inner-product search, the query first fills $M$ look-up tables $T_{m}[j]=\a{q^{(m)},c_{m,j}}$, at a cost of $K·d$ multiply-adds, after which each database score is a sum of $M$ table entries:`),
  EQ(R`\hat{s}(q,x)=\a{q,\hat{x}}=\sum_{m=1}^{M}{T_{m}\b{i_{m}(x)}}, s(q,x)−\hat{s}(q,x)=\a{q,x−\hat{x}}`, "adc"),
  P(R`The second identity is central to this report. k-means minimises the *reconstruction* error $\n{x−\hat{x}}^{2}$, but the ranking is corrupted by the *score* error $\a{q,x−\hat{x}}$, which depends on the query. Averaged over a query distribution with second-moment matrix $S=E[qq^{⊤}]$,`),
  EQ(R`E_{q}\b{\a{q,x−\hat{x}}^{2}}=(x−\hat{x})^{⊤}S(x−\hat{x})`, "scoreerr"),
  P(R`which equals the k-means objective only when $S∝I$. CLIP queries are far from isotropic, and image and text queries have different $S$, so codebooks fitted to the images are not optimal for captions. Section 3.7 exploits this. We also use **OPQ** (Ge et al., 2014), which learns an orthogonal rotation $R$ applied before quantization so that variance is balanced across the $M$ sub-spaces; it does not change inner products because $\a{Rq,Rx}=\a{q,x}$.`),

  H2("3.3", "Hierarchical navigable small world graphs (HNSW)"),
  P(R`HNSW (Malkov & Yashunin, 2020) is a proximity graph organised in layers. Each vector is assigned a maximum layer $ℓ=\floor{−ln(U)·m_{L}}$ with $U∼Uniform(0,1)$ and $m_{L}=1/ln M$, so each layer holds roughly a $1/M$ fraction of the one below it. On insertion, the vector is linked on every layer it occupies to up to $M$ near neighbours (up to $2M$ on the bottom layer, layer 0), chosen from a beam search of width $efConstruction$ by a diversity heuristic that keeps neighbours in different directions.`),
  P(R`A query enters at the single top-layer node and descends greedily: on each layer it repeatedly moves to the neighbour $v$ that maximises $s(q,v)$ until no neighbour improves on the current node. On layer 0 it runs a best-first beam search that keeps the $ef$ best candidates seen so far and stops when the best unexpanded candidate is worse than the worst kept result. The upper layers provide long-range shortcuts, giving roughly logarithmic search cost, and $ef$ trades speed for recall at query time without rebuilding. We use $M=32$ and $efConstruction=200$ unless stated otherwise. Each vector stores its full 2,048-byte embedding plus $2M$ four-byte layer-0 links, $4d+8M=2,304$ bytes, plus a small upper-layer overhead (2,320 bytes measured).`),

  H2("3.4", "Exact search"),
  P("Brute-force inner-product search (FAISS IndexFlatIP) provides the ground truth — the true top-100 of every query — and serves as the speed baseline. It needs no training and its recall is 1 by definition."),
);

// ---- 4 analysis framework and query-aware construction
add(
  P("The three methods differ in mechanism, but each ultimately decides which database vectors to return by comparing approximate or partial similarity information. The remainder of this section introduces a common model of when such decisions fail, identifies the two query properties it depends on, and derives one query-aware construction method per family from it."),

  H2("3.5", "When does an approximate index lose a neighbour?"),
  P(R`Let $s_{(r)}(q)$ denote the similarity of the query to its $r$-th true neighbour. LSH and PQ replace $s(q,x)$ with an estimate $\hat{s}(q,x)=s(q,x)+ε(q,x)$ and rank the whole database by it. A true neighbour $u∈N_{k}(q)$ is displaced by a non-neighbour $v$ whenever`),
  EQ(R`ε(q,v)−ε(q,u)>s(q,u)−s(q,v)`, "displace"),
  P(R`The right-hand side is a property of the *query*: it is small when the true top-$k$ is not well separated from the vectors just below it. We summarise this separation by the **neighbour contrast** $γ(q)=s_{(1)}(q)−s_{(100)}(q)$. If the estimation error has scale $σ_{ε}(q)$, recall is governed by the noise-to-margin ratio`),
  EQ(R`ρ(q)=\frac{σ_{ε}(q)}{γ(q)}`, "rho"),
  P(R`so a query with half the contrast behaves like a query with twice the noise. Graph search does not estimate scores — it computes them exactly — but it only computes them along a path. Its failure mode is different: the greedy step from node $u$ is productive only if some neighbour $v$ of $u$ is noticeably more similar to $q$ than $u$ is. When the query lies far from every database vector, all similarities are small and close to one another, so the walk receives little signal about which direction to take and its stopping rule fires early in the wrong region. The relevant query property is therefore **proximity to the data**, which we measure by $s_{(1)}(q)$.`),
  P("This model makes two testable predictions. **(P1)** Methods based on approximate scores (LSH, PQ) should lose recall on queries with low contrast $γ(q)$, regardless of whether the method was fitted to the data. **(P2)** Graph routing (HNSW) should lose recall on queries with low $s_{(1)}(q)$, increasingly so as the corpus — and hence the path — grows. Section 4.2 shows that text queries are worse on both counts, and Section 4.5 tests both predictions per query. The model also suggests where to intervene: reduce the score error *for the query distribution that will be served* (PQ), place the hash boundaries where the queries are (LSH), and give the graph edges that are useful from the queries' point of view (HNSW). Each intervention below uses a sample of training captions that is disjoint from the test queries, and none of them changes the exact ground truth."),

  H2("3.6", "Modality-aware centring for LSH"),
  P(R`CLIP image vectors lie in a narrow cone around $μ_{I}$. For a random hyperplane, $w^{⊤}x=w^{⊤}μ_{I}+w^{⊤}(x−μ_{I})$, and whenever $\abs{w^{⊤}μ_{I}}$ exceeds the spread of the second term, every image falls on the same side and the bit carries no information. Two standard data-dependent remedies move the hyperplanes into the cone: subtracting the image mean, $h_{i}(x)=1[w_{i}^{⊤}(x−μ_{I})>0]$, or thresholding each projection at its median over the database. Both ensure that every bit splits the images roughly in half.`),
  P(R`For a text query, however, $w^{⊤}(q−μ_{I})$ is dominated by the constant $w^{⊤}(μ_{T}−μ_{I})$, so after image-based centring the query falls on a fixed side of most hyperplanes regardless of its content. We therefore centre each modality on its own mean,`),
  EQ(R`h_{i}(x)=1\b{w_{i}^{⊤}(x−μ_{I})>0} (database), h_{i}(q)=1\b{w_{i}^{⊤}(q−μ_{T})>0} (text queries)`, "lshqa"),
  P(R`where $μ_{T}$ is estimated from the training captions. This aligns the centres of the two cones before hashing; the database codes are identical to those of image-centred LSH, so the variant costs nothing at build time.`),

  H2("3.7", "Query-weighted product quantization"),
  P(R`Eq. (${EN.scoreerr}) shows that the error that matters for a query distribution is the $S$-weighted reconstruction error. Let $A=S^{1/2}$ be the symmetric square root of the query second-moment matrix. Indexing $x'=Ax$ and searching with $q'=A^{−1}q$ leaves every inner product exactly unchanged,`),
  EQ(R`\a{q',x'}=q^{⊤}A^{−1}Ax=\a{q,x}`, "qa1"),
  P("so the ground truth is preserved, while ordinary k-means in the transformed space now minimises the expected squared score error for queries drawn from $S$:"),
  EQ(R`\n{x'−\hat{x}'}^{2}=(x−\hat{x})^{⊤}S(x−\hat{x})=E_{q}\b{\a{q,x−\hat{x}}^{2}}`, "qa2"),
  P(R`We estimate $S$ from 200,000 training captions (*text-weighted*) or as the average $\frac{1}{2}(S_{T}+S_{I})$ of the caption and image second moments (*mixed*), and regularise its eigenvalues as $λ_{j}+0.01·\bar{λ}$ so that $A^{−1}$ is well conditioned. The transform concentrates variance in a few directions, which a fixed split into contiguous sub-vectors handles poorly, so we combine it with the OPQ rotation. The extra cost is one $512×512$ matrix–vector product per query. The construction is a closed-form relative of ScaNN's anisotropic loss (Guo et al., 2020): both weight quantization error by its effect on the query score rather than by its Euclidean size.`),

  H2("3.8", "Caption-guided edges for HNSW"),
  P(R`HNSW's edges connect images that are close *to each other*. A caption's true neighbours, however, can be mutually distant images that share only their relevance to the caption — a view of a street and a close-up of a bus can both match “a red bus in a city” — so a greedy walk from far away may have no short path between them. Following RoarGraph and OOD-DiskANN, we add edges that reflect query-induced proximity. For each training caption we compute its exact top-$K$ images ($K=16$) and link its nearest image to the other $K−1$, in both directions (Algorithm 1).`),
  ...algorithm("**Algorithm 1.** Caption-guided edge insertion", [
    [0, "**Input:** HNSW graph $G$ with layer-0 capacity $2M$ per node; image vectors $X$; training captions $T$; $K=16$; per-node cap $E_{max}=16$"],
    [0, "**for each** caption $t∈T$ **do**"],
    [1, "$(u_{1},…,u_{K})←$ exact top-$K$ images of $t$ in $X$"],
    [1, "propose pairs $(u_{1},u_{j})$ and $(u_{j},u_{1})$ for $j=2,…,K$"],
    [0, "count the support $c(u,v)$ of each proposed pair; discard pairs already in $G$"],
    [0, "**for each** node $u$ **do**"],
    [1, "append $u$'s proposed neighbours to its layer-0 list in decreasing $c(u,v)$, while free slots remain and at most $E_{max}$ are added"],
    [0, "**Output:** $G'$ — same memory, same search procedure"],
  ]),
  P(R`The edges occupy layer-0 slots that HNSW has already allocated but not filled: the mean layer-0 degree is only 21.6 (COCO) and 23.1 (LAION) out of $2M=64$, so **the index size does not change**. The search algorithm is untouched. The only cost is the offline exact top-$K$ search for each training caption. To understand *why* the edges help, we also evaluate two alternative caption-aware changes as ablations (Section 4.7): starting the layer-0 search at the true neighbours of the most similar training captions (testing whether the entry point is at fault), and inserting the captions themselves into the graph as non-returnable nodes (testing whether the graph lacks landmarks near text queries).`),
);

// ---- 5 experiments
add(H1("4", "Experiments"),
  H2("4.1", "Experimental setup"),
  ...table(2, "Datasets. Both use CLIP ViT-B/32 embeddings (512-d, unit norm).",
    ["", "COCO", "LAION"],
    [["Source", "MS-COCO 2014 train + val, embedded by us", "VIBE benchmark (LAION-400M subset)"],
     ["Corpus (indexed)", "122,287 images", "999,448 images"],
     ["Image queries", "1,000 held-out images", "1,000 held-out images"],
     ["Text queries", "First caption of the same 1,000 images", "1,000 VIBE web-text queries (unpaired)"],
     ["Training captions", "122,287 captions of corpus images", "999,448 VIBE training queries"],
     ["Ground truth", "Exact top-100 by brute force", "Exact top-100 (agrees with VIBE's to ≥ 0.999)"]],
    [2, 3.5, 3.5]),
  P("**Datasets.** The two datasets differ in scale (8×) and in text style — clean human-written captions versus noisy web text — which lets us check that conclusions are not specific to one benchmark (Table 2). On COCO, the image and text query sets describe the same 1,000 held-out images, so the two sets differ only in which encoder produced the query."),
  P("**Protocol.** We follow the VIBE protocol (Jääsaari et al., 2025). (i) *Build on the corpus only*: codebooks, hyperplane statistics and graphs never see a test query; the query-aware methods of Sections 3.6–3.8 may use the training captions, never the test queries. (ii) *One index, both query types*: each index is built once and searched with image and text queries under identical parameters, so any difference is caused by the query alone. (iii) *Timing*: queries are issued one at a time on a single CPU thread pinned to one core, and we report the fastest of several passes. All runs used one server (2× AMD EPYC 7742) and FAISS 1.15.1; random seeds are fixed."),
  P(R`**Metrics.** Our main metric is recall@100 (Eq. ${EN.recall}); we also report recall@10, single-thread queries per second (QPS), index size in bytes per vector, and build time. The **modality gap in recall** is`),
  EQ(R`Δ=Recall@100(Q_{image})−Recall@100(Q_{text})`, "delta"),
  P("measured on the same index at the same setting. We never average the two query types, since their difference is the object of study."),

  H2("4.2", "Characterising the query distributions"),
  P("Before comparing indexes we verify the modality gap in our data. Table 3 shows that the mean text query points in a very different direction from the mean image (cosine 0.31–0.36), whereas held-out image queries match it almost perfectly (0.999)."),
  ...table(3, "Relationship between the queries and the indexed images. $γ$ is the neighbour contrast of Section 3.5.",
    ["Dataset", "Query type", "$cos$(query mean, image mean)", "$s_{(1)}$", "$s_{(100)}$", "Contrast $γ$"],
    [["COCO", "image", "0.999", "0.858", "0.771", "0.087"], ["COCO", "text", "0.309", "0.345", "0.299", "0.046"],
     ["LAION", "image", "0.999", "0.825", "0.733", "0.092"], ["LAION", "text", "0.357", "0.355", "0.300", "0.055"]],
    [1.2, 1.2, 2.4, 1.3, 1.3, 1.5]),
  ...figure("fig1_neighbour_similarity.png", 1720, 681, 1, "Similarity of each query to its 1st–100th true neighbour (solid: image queries; dashed: text queries; band: 10th–90th percentile)."),
  P("Figure 1 shows that text queries differ in exactly the two properties singled out in Section 3.5. They are **far from the data**: the most similar image has similarity 0.35, against 0.86 for an image query. And their neighbours are **bunched**: the contrast $γ$ between the 1st and 100th neighbour is about half that of image queries (0.046 vs 0.087 on COCO). By Eq. (" + EN.rho + "), the same absolute estimation error is therefore roughly twice as damaging for a text query."),

  H2("4.3", "Accuracy–efficiency trade-off (RQ1)"),
  P("With image queries the three families behave as their designs predict (Figures 2 and 3, Table 4)."),
  ...figure("fig2_recall_vs_qps.png", 1722, 854, 2, "Recall@100 vs single-thread queries per second. Colour denotes the method; solid lines and filled markers are image queries, dashed lines and hollow markers text queries; the star is exact search."),
  ...figure("fig3_recall_vs_memory.png", 1685, 854, 3, "Recall@100 vs index size in bytes per vector."),
  ...table(4, "Best setting of each method with image queries.",
    ["Dataset", "Method", "Setting", "Recall@100", "QPS", "Speed-up", "Bytes / vector", "Build (s)"],
    [["COCO", "HNSW", "$ef$ = 128", "0.992", "2,235", "30×", "2,320", "26"],
     ["COCO", "PQ", "$M$ = 64", "0.657", "181", "2.5×", "68", "5"],
     ["COCO", "LSH", "$b$ = 4,096", "0.837", "128", "1.7×", "581", "13"],
     ["COCO", "Exact", "—", "1", "74", "1×", "2,048", "0.2"],
     ["LAION", "HNSW", "$ef$ = 512", "0.984", "488", "51×", "2,320", "226"],
     ["LAION", "PQ", "$M$ = 64", "0.560", "22", "2.3×", "65", "12"],
     ["LAION", "LSH", "$b$ = 4,096", "0.805", "17", "1.8×", "520", "72"],
     ["LAION", "Exact", "—", "1", "9.6", "1×", "2,048", "1.3"]],
    [1.1, 1, 1.4, 1.2, 1, 1.1, 1.3, 1]),
  B("**HNSW** dominates on speed and accuracy: recall@100 ≥ 0.98 at 30–51× the speed of exact search. Its advantage grows with corpus size because it computes only 1,400–2,000 similarities per query, whereas a scan computes $n$. The price is memory — 13% *more* than the raw vectors, for the links — and build time (226 s for 1M vectors on 32 threads)."),
  B("**PQ** dominates on memory: 8–64 bytes instead of 2,048 (32–240× smaller). On its own it is neither accurate (recall@100 ≤ 0.66) nor fast, since ADC still visits every code; in practice it serves as the first stage of a pipeline with re-ranking (Section 4.7)."),
  B("**LSH** needs no training and builds in seconds, but needs 4,096 bits (512 bytes) to reach recall 0.80–0.84, at which point its full scan is barely faster than exact search."),
  P("Table 5 compares the two compression methods at equal code length. Learned PQ codebooks beat random LSH hyperplanes at every budget — slightly for image queries, but by 2.8–30× for text queries — while LSH codes are 5–10× faster to scan (a popcount instead of $M$ table look-ups). This is the classic data-dependent versus data-independent trade-off from the course, and the text-query columns already hint that data independence does not imply robustness."),
  ...table(5, "Recall@100 (image / text) at equal bits per vector; PQ uses $M=b/8$.",
    ["Bits / vector", "COCO LSH", "COCO PQ", "LAION LSH", "LAION PQ"],
    [["64", "0.155 / 0.011", "0.303 / 0.179", "0.075 / 0.002", "0.111 / 0.059"],
     ["128", "0.270 / 0.025", "0.392 / 0.229", "0.187 / 0.008", "0.198 / 0.114"],
     ["256", "0.420 / 0.071", "0.505 / 0.336", "0.337 / 0.030", "0.345 / 0.232"],
     ["512", "0.577 / 0.181", "0.657 / 0.515", "0.504 / 0.113", "0.560 / 0.438"]],
    [1.2, 2, 2, 2, 2]),

  H2("4.4", "The cost of querying with text (RQ2)"),
  P("**Every method loses recall on text queries at every setting** (Figure 4, Table 6; all gaps are many standard errors from zero), but the loss behaves differently in each family."),
  ...figure("fig4_param_sweep_gap.png", 2498, 1222, 4, "Image (solid) and text (dashed) recall@100 across each method's parameter. The shaded area is the gap Δ."),
  ...table(6, "The gap Δ = recall@100(image) − recall@100(text) at the ends of each parameter sweep.",
    ["Dataset", "Method", "Smallest setting", "Largest setting", "Largest Δ"],
    [["COCO", "HNSW", "$ef$ 100: 0.986 vs 0.884 (Δ 0.101)", "$ef$ 1024: 1.000 vs 0.996 (Δ 0.004)", "0.101"],
     ["COCO", "PQ", "$M$ 8: 0.303 vs 0.179 (Δ 0.123)", "$M$ 64: 0.657 vs 0.515 (Δ 0.142)", "0.169"],
     ["COCO", "LSH", "$b$ 64: 0.155 vs 0.011 (Δ 0.144)", "$b$ 4096: 0.837 vs 0.559 (Δ 0.279)", "**0.395**"],
     ["LAION", "HNSW", "$ef$ 100: 0.939 vs 0.749 (Δ 0.190)", "$ef$ 1024: 0.988 vs 0.956 (Δ 0.031)", "0.190"],
     ["LAION", "PQ", "$M$ 8: 0.111 vs 0.059 (Δ 0.053)", "$M$ 64: 0.560 vs 0.438 (Δ 0.122)", "0.122"],
     ["LAION", "LSH", "$b$ 64: 0.075 vs 0.002 (Δ 0.072)", "$b$ 4096: 0.805 vs 0.515 (Δ 0.290)", "**0.401**"]],
    [1.1, 1, 2.7, 2.7, 1.1]),
  B("**HNSW's gap can be bought back with search effort.** It falls from 0.10 to 0.004 on COCO and from 0.19 to 0.03 on LAION as $ef$ grows."),
  B("**PQ's gap does not shrink with longer codes**: it stays at 0.12–0.17 on COCO and grows to 0.12 on LAION."),
  B("**LSH has the largest gap of all**, up to 0.40. At 512 bits, text queries retain only 22–31% of the image-query recall."),
  P("The LSH result is the most informative. LSH learns nothing from the images, so its loss cannot be caused by an index fitted to the wrong distribution; it must originate in the text queries themselves. This is the first sign that the modality penalty is, at least in part, a property of the query distribution rather than of the index."),
  P("To express the gap as a cost, Table 7 reports the search effort HNSW needs for text queries to reach the recall that image queries obtain at a cheap setting."),
  ...table(7, "The “text tax” for HNSW: search effort needed for text queries to match image-query recall.",
    ["Dataset", "Image queries: $ef$ → recall (QPS)", "Text queries need $ef$", "Text QPS", "Slow-down"],
    [["COCO", "100 → 0.986 (2,700)", "≈ 490", "≈ 790", "**3.4×**"],
     ["COCO", "128 → 0.992 (2,235)", "≈ 700", "≈ 580", "**3.9×**"],
     ["LAION", "100 → 0.939 (1,870)", "≈ 650", "≈ 400", "**4.7×**"],
     ["LAION", "128 → 0.952 (1,553)", "≈ 900", "≈ 295", "**5.3×**"]],
    [1.1, 3, 1.8, 1.4, 1.4]),
  P("An engine tuned on image queries and then served captions would deliver 10–20 points less recall than its benchmark promised, and recovering it would cost 3–5× more search time."),

  H2("4.5", "Testing the score-margin model (RQ3)"),
  P("We now test the two predictions of Section 3.5 query by query (Figure 5)."),
  ...figure("fig5_which_queries_fail.png", 1982, 1262, 5, "Per-query recall@100, with queries grouped into 12 equal-size bins. Left: HNSW against the query's similarity to its nearest image, $s_{(1)}$. Right: PQ against the neighbour contrast $γ$."),
  P(R`**(P1) Low contrast defeats approximate scoring.** For PQ, per-query recall rises with the neighbour contrast (Spearman $ρ$ up to +0.51), and 39% of COCO text queries — against 3% of image queries — obtain recall below 0.5. Notably, the absolute PQ score error is *smaller* for text queries (RMS 0.038 vs 0.052 on COCO at $M=64$); what differs is the margin, so the noise-to-margin ratio of Eq. (${EN.rho}) is 0.78 for text against 0.57 for images.`),
  P(R`For LSH the prediction can be made quantitative. We first verified that the empirical per-bit collision rate matches Charikar's $1−θ/π$ to within 0.001 for *both* modalities: the hash functions treat a text pair and an image pair at the same angle identically. The difference lies entirely in the margins. On LAION, the median angular margin between a query's 100th and 1,000th neighbours is 4.7° for image queries but only 1.8° for text queries. Table 8 compares the swap probability predicted by Eq. (${EN.swap}) with the rate measured from the actual codes.`),
  ...table(8, "LSH on LAION: probability that the 1,000th neighbour outranks the 100th, predicted by the binomial model (Eq. " + EN.swap + ") and measured (seed 0).",
    ["Query type", "Margin", "$b$", "$z$", "Predicted", "Measured", "Recall@100"],
    [["image", "4.7°", "256", "0.82", "0.199", "0.169", "0.337"],
     ["image", "4.7°", "512", "1.16", "0.126", "0.104", "0.504"],
     ["image", "4.7°", "4,096", "3.28", "0.006", "0.007", "0.805"],
     ["text", "1.8°", "256", "0.30", "0.380", "0.349", "0.030"],
     ["text", "1.8°", "512", "0.42", "0.333", "0.333", "0.113"],
     ["text", "1.8°", "4,096", "1.19", "0.123", "0.091", "0.517"]],
    [1.2, 1, 1, 1, 1.2, 1.2, 1.3], "The model is slightly pessimistic because rows of a random rotation are orthogonal rather than independent."),
  P(R`The model tracks the measurements closely. It also explains the size of the LSH gap: since $z∝δ\sqrt{b}$, a margin 2.6× smaller requires $2.6^{2}≈7$ times more bits for the same separation. Indeed, text queries at 4,096 bits ($z=1.19$, recall 0.517) behave almost exactly like image queries at 512 bits ($z=1.16$, recall 0.504) — eight times the code length, or three doublings, which is the horizontal offset between the LSH curves in Figure 4. The data-independent method suffers most precisely *because* it is data-independent: unlike PQ, nothing in its construction can adapt to where the queries' margins are small.`),
  P(R`**(P2) Distance from the data misleads graph routing.** HNSW's per-query recall rises with the query's similarity to its nearest image (Spearman $ρ=+0.55$ on COCO, +0.64 on LAION; Figure 5, left), and text queries occupy the far end of this axis. Strikingly, on LAION at $ef=100$ text queries perform 1% *fewer* similarity computations than image queries yet lose 19 points of recall (Figure 6): the beam search's stopping rule is satisfied in the wrong region, exactly as the flat-similarity argument of Section 3.5 predicts.`),
  ...figure("fig6_hnsw_extra_work.png", 990, 638, 6, "Extra similarity computations per text query relative to an image query (HNSW).", 3.6),

  H2("4.6", "Ablation studies"),
  H3("HNSW build parameters"),
  P("Figure 7 varies HNSW's construction parameters on COCO. More links per node shrink the gap: from $M=8$ to $M=64$, text recall at $ef=100$ rises from 0.79 to 0.89 and Δ falls from 0.155 to 0.098, for 21% more memory, with most of the gain reached by $M=16$–32. $efConstruction$ has no consistent effect (text recall 0.92, 0.86, 0.88 and 0.90 for 40, 100, 200 and 400). Repeated builds with the same seed are identical, so these differences are real but follow no trend: a more thorough construction optimises the graph for image-to-image proximity, which is not what text queries need."),
  ...figure("fig7_hnsw_build_params.png", 1701, 674, 7, "HNSW recall@100 against graph degree $M$ (left) and $efConstruction$ (right), COCO."),
  H3("Corpus size"),
  P("To separate the effect of scale from that of the dataset, Figure 8 evaluates random LAION subsets with the ground truth recomputed for each. HNSW's gap grows steadily with $n$: at $ef=100$ it is 0.10 at 10k images, 0.13 at 100k and 0.19 at 1M, consistent with prediction (P2) that a longer greedy path offers more opportunities to stall in the wrong region. PQ's gap does not grow (0.10–0.12 at every size). HNSW's text penalty is thus a *scale* problem, whereas PQ's is a fixed property of its codes."),
  ...figure("fig8_scaling.png", 1701, 717, 8, "Recall@100 against corpus size (random LAION subsets). Left: HNSW; right: PQ."),
  H3("Removing the modality offset"),
  P(R`If the loss arises because text and images occupy different regions, removing the offset between them should reduce it. As a diagnostic, we subtract $μ_{I}$ from the images and $μ_{T}$ from the text queries, re-normalise, and rebuild every index *and the ground truth* in the centred space (Table 9).`),
  ...table(9, "Effect of centring each modality on its own mean (recall@100 image / text, and Δ).",
    ["Dataset", "Method", "Original space", "Centred space"],
    [["COCO", "HNSW, $ef$ = 100", "0.986 / 0.884 (Δ 0.101)", "0.987 / 0.931 (**Δ 0.057**)"],
     ["COCO", "PQ, $M$ = 64", "0.656 / 0.512 (Δ 0.143)", "0.738 / 0.573 (Δ 0.165)"],
     ["LAION", "HNSW, $ef$ = 100", "0.939 / 0.749 (Δ 0.190)", "0.964 / 0.868 (**Δ 0.096**)"],
     ["LAION", "PQ, $M$ = 64", "0.561 / 0.437 (Δ 0.124)", "0.678 / 0.514 (Δ 0.164)"]],
    [1.1, 2, 3, 3]),
  P("Centring **halves HNSW's gap**, supporting (P2). For PQ it roughly doubles the contrast of *both* query types (COCO: 0.087 → 0.168 for images, 0.046 → 0.090 for text), so both recalls rise; but text queries still have half the image contrast and the gap remains, supporting (P1). Centring is not a usable improvement in itself, however, because it changes which images count as nearest: only 62–72% of each text query's original top-100 survives. The query-aware methods evaluated next keep the original ground truth."),

  H2("4.7", "Evaluating query-aware index construction"),
  P("All results in this section are measured against the original exact top-100, so they compare directly with Sections 4.3–4.5."),
  H3("HNSW with caption-guided edges"),
  ...table(10, "HNSW variants at $ef$ = 100 (recall@100). Variants 1 and 2 are the ablations described in Section 3.8.",
    ["Dataset", "Variant", "Image", "Text", "Δ", "Extra memory"],
    [["COCO", "Baseline", "0.986", "0.884", "0.101", "—"],
     ["COCO", "1. Caption entry points", "0.986", "0.877", "0.108", "+2,328 B/vector"],
     ["COCO", "2. Captions as graph nodes", "0.985", "**0.078**", "0.907", "+232 B/vector"],
     ["COCO", "3. Caption-guided edges (Alg. 1)", "**0.987**", "**0.947**", "**0.040**", "none"],
     ["LAION", "Baseline", "0.939", "0.749", "0.190", "—"],
     ["LAION", "3. Caption-guided edges (Alg. 1)", "**0.950**", "**0.818**", "**0.132**", "none"]],
    [1.1, 3.2, 1, 1, 1, 1.7]),
  ...figure("fig9_hnsw_fixes.png", 1741, 879, 9, "HNSW variants: recall@100 against QPS (dashed: text queries). Purple: caption-guided edges (lighter shades use fewer training captions); green: caption entry points."),
  P("Table 10 and Figure 9 separate the three candidate explanations for HNSW's text penalty."),
  B("**A better starting point does not help** (variant 1): the descent through the upper layers is not where the search fails."),
  B("**Captions as nodes are harmful** (variant 2): a caption node is far more similar to a text query (cosine ≈ 0.8) than any image (≈ 0.3), so caption nodes fill the entire beam and images are never reached. This is why RoarGraph uses queries to shape edges rather than as nodes."),
  B("**Caption-guided edges work** (variant 3). On COCO, text recall rises from 0.88 to 0.95, Δ falls by 60%, and image recall even improves slightly. The gain holds at equal speed: plain HNSW reaches text recall 0.948 at about 1,470 QPS, while the edited graph reaches 0.947 at about 2,170 QPS, **1.5× faster**. On LAION the benefit grows with the number of training captions used — none at equal speed with 10% of them, and with all of them +7 points at $ef=100$ and 3–6 points at equal speed."),
  P("The construction cost is an offline exact top-16 search per training caption (73 s for COCO; 1.7 h on 32 cores for LAION's 1M captions), which adds 6.6 (COCO) and 11.0 (LAION) edges per node on average. The pattern confirms the model: text queries fail not because the search starts in the wrong place, but because the image graph lacks shortcuts between images that are jointly relevant to a caption without being close to each other."),

  H3("Query-weighted PQ"),
  P("Table 11 crosses the two components of Section 3.7 — query weighting and the OPQ rotation — and adds the mixed weighting. The last two columns report the noise-to-margin ratio $ρ$ of Eq. (" + EN.rho + ") at $M$ = 64, computed from the RMS score error over each query's true top-100."),
  ...table(11, "PQ variants at $M$ = 64 bytes per vector: recall@100 (image / text, Δ) and noise-to-margin ratio $ρ$ (image / text).",
    ["Variant", "Weights", "OPQ", "COCO recall", "LAION recall", "COCO $ρ$", "LAION $ρ$"],
    [["PQ (baseline)", "none", "no", "0.656 / 0.512 (Δ 0.143)", "0.561 / 0.437 (Δ 0.124)", "0.57 / 0.78", "0.72 / 0.82"],
     ["OPQ", "none", "yes", "0.773 / 0.665 (Δ 0.108)", "0.662 / 0.537 (Δ 0.124)", "0.26 / 0.36", "0.47 / 0.56"],
     ["Text-weighted PQ", "captions", "no", "0.160 / 0.461", "0.124 / 0.392", "0.99 / 0.82", "0.95 / 0.90"],
     ["**Text-weighted OPQ**", "captions", "yes", "0.632 / **0.793** (Δ −0.161)", "0.554 / **0.579** (Δ −0.025)", "0.43 / **0.21**", "0.65 / **0.53**"],
     ["**Mixed OPQ**", "50/50", "yes", "**0.809** / 0.732 (Δ 0.077)", "**0.691** / 0.519 (Δ 0.171)", "0.26 / 0.26", "0.50 / 0.60"]],
    [1.6, 1.0, 0.75, 2.35, 2.35, 1.1, 1.1]),
  ...figure("fig10_pq_fixes.png", 1983, 1302, 10, "PQ variants: recall@100 against code size, for image queries (top) and text queries (bottom)."),
  B("**The rotation is essential.** Weighting alone concentrates variance in a few directions that a contiguous split quantizes badly; image recall collapses to 0.12–0.16 and $ρ$ approaches 1."),
  B("**The gap is a design choice, not a law.** Text-weighted OPQ raises COCO text recall from 0.51 to 0.79 (+55%) at the same 64 bytes and reverses the sign of the gap; on LAION it removes the gap at every code size."),
  B("**A mixture serves both query types.** Mixed OPQ gives the best image recall and the second-best text recall on COCO, halving Δ."),
  B("**The mechanism is visible in ρ.** On COCO the ratio orders the variants almost exactly as recall does, and mixed weighting equalises it across modalities (0.26 vs 0.26), which is when the gap is smallest. On LAION the ordering holds for text queries and broadly for images."),
  P("**Re-ranking.** A common alternative is to re-score PQ's top candidates with the full vectors. Re-ranking the top 1,000 (at most 21% slower) lifts $M$ = 64 recall to 0.998 / 0.974 on COCO and 0.968 / 0.895 on LAION, but requires the full vectors in RAM or on SSD — giving up PQ's memory advantage — and cannot recover neighbours missing from the candidate list. With short codes the gap even widens as the list grows (LAION, $M$ = 16: Δ 0.12 → 0.20 from 200 to 1,000 candidates). Query-aware codes and re-ranking are therefore complementary: better codes put more true neighbours into the list that re-ranking sorts."),

  H3("Modality-aware LSH"),
  ...figure("fig11_lsh_variants.png", 1983, 1302, 11, "LSH variants: recall@100 against code length. Top: image queries (the three data-dependent variants coincide). Bottom: text queries."),
  P("Figure 11 compares plain LSH with the three variants of Section 3.6."),
  B("**For image queries, balanced bits help when bits are scarce** (COCO, 128 bits: 0.27 → 0.35), although plain LSH is slightly better at 4,096 bits."),
  B(R`**For text queries, image-based adjustments are harmful**: median thresholds and image-mean centring reduce text recall at 4,096 bits to 0.22 (COCO) and 0.07 (LAION), from 0.56 and 0.52. Every hyperplane now passes through the image cone, so a text query, lying outside it, falls on the same side of most hyperplanes and its code barely varies — the failure anticipated in Section 3.6.`),
  B("**Modality-aware centring (Eq. " + EN.lshqa + ") recovers and exceeds the baseline**: text recall roughly doubles at mid-range budgets (COCO, 512 bits: 0.18 → 0.33; LAION, 1,024 bits: 0.23 → 0.40), and the LAION gap at 4,096 bits falls from 0.29 to 0.12."),

  H2("4.8", "Success and failure cases"),
  P("Figure 12 shows three COCO test captions, chosen to illustrate the two failure modes, with the exact top-10 images and each method's top-10."),
  ...figure("fig12_examples.png", 1722, 1257, 12, "Three COCO test captions. Rows: exact top-10, HNSW ($ef$ = 100), HNSW with caption-guided edges, PQ ($M$ = 64). Green border: in the exact top-10; red: not."),
  B("**HNSW stalls in a neighbouring region.** For the long train caption it drifts into lorries and car parks (recall@10 = 0.5); with caption-guided edges it returns exactly the true top-10."),
  B("**PQ finds the topic but not the order.** For the baseball caption every PQ result is a plausible baseball photo, yet none is in the exact top-10: hundreds of near-identical scores are reshuffled by small errors — low contrast in action. The example also shows a limitation of recall against exact search, since a user might well be satisfied with PQ's answer."),
  B("**Success is the norm for HNSW.** Averaged over all captions, text recall@10 is 0.949 for plain HNSW and 0.987 with caption-guided edges."),
);

// ---- 6 discussion
add(H1("5", "Discussion"),
  H2("5.1", "Strengths and weaknesses"),
  ...table(12, "Summary of the three methods, from our measurements.",
    ["Method", "Strengths", "Weaknesses", "Best use"],
    [["HNSW", "Recall@100 0.98–0.99 at 30–51× exact speed; tunable at query time; text loss shrinks with $ef$ and caption-guided edges", "More memory than the raw vectors; slow build; text penalty grows with corpus size", "Latency-critical serving with sufficient RAM"],
     ["PQ", "32–240× smaller; fast build; query-weighted OPQ recovers text recall", "Recall ≤ 0.66 alone; flat scan barely faster than exact; gap does not shrink with longer codes", "Memory-bound systems, as a first stage before re-ranking"],
     ["LSH", "No training; builds in seconds; 5–10× faster scan than PQ at equal bits; well-understood error model", "Needs 4,096 bits for recall 0.8; largest text gap (up to 0.40); image-fitted variants collapse text recall", "Streaming data and quick baselines, with modality-aware centring"]],
    [1, 3.2, 3.2, 2.2]),
  H2("5.2", "Key factors affecting performance"),
  P("The experiments isolate two mechanisms behind the modality penalty. **Low neighbour contrast** is a property of the queries, not of the index: it appears even in LSH, which learns nothing, and extra code length reduces it only slowly ($z∝δ\\sqrt{b}$). What removes it is spending the code's precision where the queries need it — query-weighted codebooks or modality-aware hash boundaries. **Distance from the data** is specific to graph search: it grows with corpus size, is halved by centring, and is remedied not by a better starting point but by edges that connect images relevant to the same caption. Both mechanisms point to one principle: **an ANN index should be constructed for the distribution of queries it will serve, not only for the data it stores.**"),
  P("Within each method, the parameters behave as follows. For LSH, code length $b$ is the only lever and its returns diminish as $1/\\sqrt{b}$. For PQ, $M$ sets the memory–accuracy trade-off, but the choice of objective (rotation and weighting) matters more for text queries than doubling $M$. For HNSW, $ef$ is the most effective query-time lever, $M$ helps up to about 32, and $efConstruction$ has no reliable effect on text recall."),
  H2("5.3", "Practical recommendations"),
  B("**Tune and evaluate on the queries that will be served.** An HNSW index tuned to 0.94–0.99 recall on image queries delivers 0.75–0.88 on captions; matching it costs 3–5× more search time."),
  B("**If a sample of real queries exists, use it at build time.** Caption-guided edges cost no memory; query-weighted OPQ costs one $512×512$ matrix."),
  B("**Do not centre or threshold with image statistics alone** when queries come from another modality."),
  H2("5.4", "Limitations"),
  B("We use one encoder (CLIP ViT-B/32); newer encoders may have a smaller modality gap."),
  B("Timing is single-threaded on CPU. To finish in time, independent jobs ran concurrently on separate cores of one server; this may lower absolute QPS (mainly for memory-bound scans on LAION) but never affects recall."),
  B("Recall is measured against exact search rather than human relevance, so an approximate “miss” may still be a relevant image (Figure 12)."),
  B("The query-aware methods require a sample of training queries from the same distribution as the real ones; we did not study robustness to a shift between the two."),
);

// ---- 7 conclusion
add(H1("6", "Conclusion"),
  P("We compared hashing (LSH), quantization (PQ) and graph (HNSW) indexes for text-to-image search on CLIP embeddings, on COCO (122k images) and LAION (1M images), searching each index with both image and text queries. With image queries the classic trade-off holds: HNSW is the fastest and most accurate, PQ the most compact, and LSH the cheapest to build. With text queries every method loses recall at every setting — the data-independent LSH most of all."),
  P("A score-margin model explains these results. Text queries' true neighbours are bunched together, which magnifies the error of approximate scoring (LSH, PQ); for LSH a binomial analysis reproduces the measured swap rates and the eight-fold increase in code length that text queries require. Text queries also lie far from every image, which weakens greedy graph routing (HNSW) increasingly as the corpus grows. The model led to three query-aware construction methods that keep the exact ground truth: caption-guided edges in HNSW's unused slots (COCO text recall 0.88 → 0.95, no extra memory), query-weighted OPQ (PQ text recall 0.51 → 0.79 at 64 bytes), and modality-aware LSH centring (text recall roughly doubled at mid-range budgets)."),
  P("The broader lesson is that cross-modal search indexes should be built, tuned and evaluated with the queries they will actually serve. Natural next steps are a GPU pipeline for caption-guided edges at web scale, combining the graph and quantization methods in IVF-PQ or disk-based graph indexes, and repeating the study with newer encoders."),
);

// ---- references
add(H1("", "References"),
  ...[
    "M. Aumüller, E. Bernhardsson, A. Faithfull. ANN-Benchmarks: a benchmarking tool for approximate nearest neighbor algorithms. Information Systems 87, 2020.",
    "M. S. Charikar. Similarity estimation techniques from rounding algorithms. STOC, 2002.",
    "M. Chen et al. RoarGraph: a projected bipartite graph for efficient cross-modal approximate nearest neighbor search. VLDB, 2024.",
    "M. Douze et al. The Faiss library. arXiv:2401.08281, 2024.",
    "T. Ge, K. He, Q. Ke, J. Sun. Optimized product quantization. IEEE TPAMI 36(4), 2014.",
    "R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, S. Kumar. Accelerating large-scale inference with anisotropic vector quantization. ICML, 2020.",
    "E. Jääsaari et al. VIBE: Vector Index Benchmark for Embeddings. arXiv:2505.17810, 2025.",
    "S. Jaiswal et al. OOD-DiskANN: efficient and scalable graph ANNS for out-of-distribution queries. arXiv:2211.12850, 2022.",
    "H. Jégou, M. Douze, C. Schmid. Product quantization for nearest neighbor search. IEEE TPAMI 33(1), 2011.",
    "W. Liang, Y. Zhang, Y. Kwon, S. Yeung, J. Zou. Mind the gap: understanding the modality gap in multi-modal contrastive representation learning. NeurIPS, 2022.",
    "T.-Y. Lin et al. Microsoft COCO: common objects in context. ECCV, 2014.",
    "Y. A. Malkov, D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI 42(4), 2020.",
    "A. Radford et al. Learning transferable visual models from natural language supervision. ICML, 2021.",
    "C. Schuhmann et al. LAION-400M: open dataset of CLIP-filtered 400 million image-text pairs. arXiv:2111.02114, 2021.",
  ].map((r) => new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(r, { size: 20 }),
    spacing: { after: 60, line: 240 }, alignment: AlignmentType.JUSTIFIED })),
);

// ================================================================== DOCUMENT
const numbered = (ref, text) => ({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text,
  alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 420 } } } }] });
const small = (children) => new Paragraph({ alignment: AlignmentType.CENTER, children });
const doc = new Document({
  creator: "SC4020 Group XX",
  title: "Hashing, Quantization and Graph Indexes for Cross-Modal Nearest Neighbour Search",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 280 } } } }] },
    numbered("nums", "%1."), numbered("refs", "[%1]"),
    { reference: "rq", levels: [{ level: 0, format: LevelFormat.NONE, text: "", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 0 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [small([new TextRun({ text: "SC4020 Technical Review — Cross-Modal Nearest Neighbour Search", size: 17, italics: true, color: "555555" })])] }) },
    footers: { default: new Footer({ children: [small([new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "555555" })])] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
