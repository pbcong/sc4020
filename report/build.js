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
  const re = /(\$[^$]+\$|\*\*__[^_]+__\*\*|__[^_]+__|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("$")) out.push(math(t.slice(1, -1)));
    else if (t.startsWith("__")) out.push(new TextRun({ text: t.slice(2, -2), underline: {}, ...base }));
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
    children: [new TextRun({ text: "SC4020 Project 1: Technical Review (Similarity Search), Nanyang Technological University", size: 20, italics: true })] }),
);

// ---- abstract
const ABS = { left: 567, right: 567 };
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Abstract", bold: true, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: ABS, spacing: { after: 100, line: 252 }, children: runs(
    "Text-to-image search engines usually embed captions and photos with a model such as CLIP and then search the image vectors with an approximate nearest neighbour (ANN) index. The index is built from the images, but the queries are captions, and CLIP puts text and image vectors in different parts of the space (the *modality gap*). We compare three ANN methods from different families, random-projection LSH (hashing), Product Quantization (quantization) and HNSW (graph), on MS-COCO (122k images) and LAION (1M images). Each index is searched with both image and text queries using the same settings. All three methods lose recall on text queries at every setting we tried. We expected LSH to be unaffected because it does not learn from the data, but it loses the most (up to 0.40 recall@100), compared with 0.17 for PQ and 0.19 for HNSW. We trace this to two properties of text queries. Their true neighbours have very similar scores, so small scoring errors in LSH and PQ reorder them, and they are far from every image, which makes HNSW's greedy search stop in the wrong place, more so on larger corpora. For LSH, a simple binomial model predicts the measured loss well. We then improve each baseline for text queries without changing the ground truth. Adding caption-guided edges to HNSW raises COCO text recall from 0.88 to 0.95 with no extra memory, query-weighted OPQ raises PQ text recall from 0.51 to 0.79 at 64 bytes per vector, and centring each modality separately roughly doubles LSH text recall at medium code lengths.", { size: 22 }) }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: ABS, spacing: { after: 280 }, children: [
    new TextRun({ text: "Keywords: ", bold: true, size: 22 }),
    new TextRun({ text: "approximate nearest neighbour search, locality-sensitive hashing, product quantization, HNSW, cross-modal retrieval, CLIP, out-of-distribution queries", italics: true, size: 22 })] }),
);

// ---- 1 introduction
add(H1("1", "Introduction"),
  P("Similarity search means finding the $k$ items in a database that are most similar to a query. It is used in image search, recommendation and retrieval-augmented generation. The exact answer comes from a linear scan that compares the query with every stored vector, which costs $O(nd)$ for $n$ vectors of dimension $d$. On our LAION corpus of one million images, a single-threaded scan takes 104 ms per query. Approximate nearest neighbour (ANN) indexes accept a small loss in accuracy in return for much faster search or much less memory. They either make each comparison cheaper, using compact codes as in LSH and PQ, or make fewer comparisons, by partitioning the data or navigating a graph."),
  P("Models such as CLIP (Radford et al., 2021) allow the same indexes to be used for *cross-modal* search. A caption and a photo are mapped into the same 512-dimensional space, and the photos closest to the caption are returned. However, the two modalities do not mix in this space. Each sits in its own narrow cone, and the two cones are far apart (Liang et al., 2022). In our data a caption and its own photo have a cosine similarity of only about 0.3. So an index whose structure was learned from image vectors gets queries from a region it has never seen. The VIBE benchmark (Jääsaari et al., 2026) shows that the throughput of general-purpose ANN methods drops sharply on such out-of-distribution (OOD) queries, and explains this briefly: an OOD query can be far from the corpus, and its nearest neighbours can be far from the query and from each other. We look at this more closely, with a controlled comparison across three index families and measurements for each query."),
  P("To study this, we keep the index, its parameters and the ground truth fixed and change only the query modality. We pick one method from each major ANN family: random-projection **LSH** (hashing, does not learn from data) and **Product Quantization** (quantization, learns codebooks from data), which are both covered in the course, and **HNSW** (graph, learns a neighbourhood graph from data), which is outside the syllabus and is widely used in vector databases such as pgvector, Milvus and Weaviate. Before running the experiments we expected PQ, whose codebooks are fitted to the images, to suffer most from text queries, and LSH, which learns nothing from the data, to suffer least. The results turned out the other way round, and explaining why became the main thread of the report. In this report we:"),
  B("**Benchmark** the three methods and exact search on COCO (122k images) and LAION (1M images). We sweep each method's main parameter and search every index with both image and text queries, comparing recall, speed, memory and build time."),
  B("**Analyse** why text queries lose recall, by measuring how they differ from image queries and how each query's recall relates to those differences. For LSH, a simple probabilistic model predicts the measured loss."),
  B("**Run ablations** on HNSW's build parameters, the corpus size and the offset between the two modalities."),
  B("**Improve the baselines** for text queries using a sample of training captions, adapting ideas from earlier work: modality-aware centring for LSH, query-weighted OPQ for PQ, and caption-guided edges for HNSW. None of these changes the search algorithm or the exact ground truth."),
  P("Section 2 covers the search problem and the modality gap. Section 3 describes the three methods using the course notation, a simple model of when they fail on text queries, and the improvements we test. Section 4 presents the experiments, Section 5 discusses strengths, weaknesses and limitations, and Section 6 concludes."),
);

// ---- 2 background
add(H1("2", "Background and Related Work"),
  H2("2.1", "Nearest neighbour search"),
  P("Let $X=\\c{x_{1},…,x_{n}}⊂ℝ^{d}$ be the database and $q∈ℝ^{d}$ a query. For a similarity function $s(q,x)$, the exact $k$-nearest-neighbour set is"),
  EQ(R`N_{k}(q)=\lim{argmax}{S⊂X, \abs{S}=k} \sum_{x∈S}{s(q,x)}`, "knn"),
  P("All our vectors are L2-normalised, so the inner product, cosine similarity and Euclidean distance give the same ranking:"),
  EQ(R`\n{q−x}^{2}=\n{q}^{2}+\n{x}^{2}−2\a{q,x}=2−2\a{q,x}=2−2 cos θ(q,x)`, "equiv"),
  P(R`where $θ(q,x)$ is the angle between the two vectors. We therefore use $s(q,x)=\a{q,x}$ throughout. An ANN method returns an approximate set $\tilde{N}_{k}(q)$, and we measure its quality by recall against the exact answer:`),
  EQ(R`Recall@k=\frac{1}{\abs{Q}}\sum_{q∈Q}{\frac{\abs{\tilde{N}_{k}(q)∩N_{k}(q)}}{k}}`, "recall"),
  P("where $Q$ is the set of queries. Recall measures how closely the index matches exact search, not whether a person would find the results relevant. We do not use a human-relevance metric because it would mix the index's errors with CLIP's own errors."),

  H2("2.2", "Contrastive embeddings and the modality gap"),
  P(R`CLIP trains an image encoder $f_{I}$ and a text encoder $f_{T}$ on image–caption pairs with a symmetric InfoNCE loss. For a batch of $B$ pairs $(i_{j},t_{j})$ and temperature $τ$, the image-to-text part of the loss is`),
  EQ(R`L_{I→T}=−\frac{1}{B}\sum_{j=1}^{B}{log\frac{exp(\a{f_{I}(i_{j}),f_{T}(t_{j})}/τ)}{\sum_{l=1}^{B}{exp(\a{f_{I}(i_{j}),f_{T}(t_{l})}/τ)}}}`, "clip"),
  P(R`The loss only requires a matching pair to be *more* similar than the non-matching pairs. Nothing in it asks the image and text vectors to overlap. Liang et al. (2022) show that each encoder maps its inputs into a narrow cone, an effect that is already present at random initialisation, and that contrastive training then keeps the two cones apart. With $μ_{I}=E[f_{I}(i)]$ and $μ_{T}=E[f_{T}(t)]$ as the mean image and text vectors, we measure the gap by $cos(μ_{I},μ_{T})$. It is 0.31 on COCO and 0.36 on LAION (Section 4.2), so for an index built on images, a text query is an out-of-distribution query.`),

  H2("2.3", "Related work"),
  P("**ANN benchmarks.** We follow the recall versus throughput methodology of ann-benchmarks (Aumüller et al., 2020) and the FAISS library (Douze et al., 2024). VIBE (Jääsaari et al., 2026) adds newer embedding datasets, including LAION with text queries, and finds that general-purpose methods, graph indexes included, need much more time to reach the same recall on these queries. We use its protocol and its LAION data, and we extend the comparison to hashing and quantization and look at the cause of the loss."),
  P("**Graph indexes for OOD queries.** RoarGraph (Chen et al., 2024) and OOD-DiskANN (Jaiswal et al., 2022) use a sample of training queries to add or change graph edges so that search works for queries outside the data distribution. Our caption-guided edges (Section 3.8) are a simple version of this idea that adds edges to an existing HNSW graph without rebuilding it."),
  P("**Query-aware quantization.** OPQ (Ge et al., 2014) learns a rotation of the data before PQ to reduce quantization error, which in practice spreads the variance more evenly across the sub-spaces. ScaNN (Guo et al., 2020) points out that for inner-product search, reconstruction error is not the right objective, because what matters is the error in the inner products with the queries. Its anisotropic loss penalises the part of the error parallel to the data point more than the orthogonal part. Our query-weighted PQ (Section 3.7) is a closed-form variant that weights the error by the second-moment matrix of the queries."),
);

// ---- 3 methods
add(H1("3", "Methods"),
  P("Sections 3.1 to 3.4 describe the three methods and the exact baseline. Section 3.5 gives a simple model of why they lose recall on out-of-distribution queries, and Sections 3.6 to 3.8 describe the improvements we derive from it. Table 1 lists the storage and query cost of each method and the parameter we sweep."),
  ...table(1, "The compared methods. $n$: database size, $d$: dimension (512), $b$: LSH code length in bits, $M$: number of PQ sub-spaces or HNSW links per node, $K$: centroids per sub-space (256), $ef$: HNSW search beam width.",
    ["Method", "Family", "Learns from data", "Bytes per vector", "Query cost", "Parameter swept"],
    [["LSH (Charikar, 2002)", "Hashing", "No", "b / 8", "O(nb) bit ops", "b = 64 … 4,096"],
     ["PQ (Jégou et al., 2011)", "Quantization", "Yes (k-means)", "M", "O(Kd + nM)", "M = 8, 16, 32, 64"],
     ["HNSW (Malkov & Yashunin, 2020)", "Graph", "Yes (graph)", "4d + 8M", "≈ O(ef·M·d·log n)", "ef = 100 … 1,024"],
     ["Exact (linear scan)", "-", "-", "4d", "O(nd)", "-"]],
    [2.6, 1.3, 1.4, 1.3, 1.9, 1.9]),

  H2("3.1", "Locality-sensitive hashing by random projection"),
  P(R`As in the course, we draw $b$ random vectors $w_{1},…,w_{b}$ from a standard Gaussian and define one binary hash function for each ($1[·]$ is 1 if the condition holds and 0 otherwise):`),
  EQ(R`h_{i}(x)=1\b{w_{i}^{⊤}x>0}, H(x)=(h_{1}(x),…,h_{b}(x))∈\c{0,1}^{b}`, "hash"),
  P(R`Each $w_{i}$ is the normal of a hyperplane through the origin, and $h_{i}$ records which side of that hyperplane a vector is on. Two vectors are compared by the Hamming distance between their codes, $D_{H}(q,x)=popcount(H(q) ⊕ H(x))$, which takes one XOR and one population count per 64 bits. Charikar (2002) showed that a random hyperplane separates two vectors with probability proportional to the angle between them:`),
  EQ(R`Pr\b{h_{i}(q)≠h_{i}(x)}=\frac{θ(q,x)}{π} ⇒ E\b{D_{H}(q,x)}=\frac{b θ(q,x)}{π}`, "charikar"),
  P(R`So the Hamming distance is an unbiased estimate of the angle, and through Eq. (${EN.equiv}) of the inner product. The bits are independent Bernoulli trials, so $D_{H}$ follows a binomial distribution and the estimate gets more accurate as $b$ grows. This is why, as noted in the lecture, more hash functions give more accurate top-$k$ results.`),
  P(R`The course covers two ways to search with LSH codes: a linear scan over all codes, or hash tables whose buckets are visited in order of Hamming distance. We use the linear scan (FAISS IndexLSH), which ranks every database code by $D_{H}$ and gives the full top-100 list needed to measure recall. A code takes $b/8$ bytes, so a 512-bit code (64 bytes) is 32 times smaller than the 2,048-byte float vector. The hyperplanes are the rows of a seeded random rotation, and every LSH result is averaged over three seeds. The standard deviation of recall across seeds is below 0.01.`),
  P(R`**How precise is the estimate?** Recall depends on whether a true neighbour $u$ (for example the 100th) stays ahead of a non-neighbour $v$, where $θ(q,u)<θ(q,v)$. For each hyperplane let $D_{i}=1[h_{i}(q)≠h_{i}(v)]−1[h_{i}(q)≠h_{i}(u)]$. Its mean is the angular margin $δ=(θ(q,v)−θ(q,u))/π$. Since $D_{i}≠0$ exactly when the hyperplane separates $u$ from $v$, its variance is $θ(u,v)/π−δ^{2}$. Adding up $b$ independent hyperplanes, the probability that the codes rank $v$ above $u$ is approximately`),
  EQ(R`Pr[swap]≈Φ(−z), z=δ\sqrt{\frac{b}{θ(u,v)/π−δ^{2}}}`, "swap"),
  P(R`where $Φ$ is the standard normal CDF. Because $z$ grows with $δ\sqrt{b}$, halving the angular margin means four times as many bits are needed for the same swap probability. Nothing here depends on where the data are, which is why we first expected LSH to behave the same for both query types. Section 3.5 explains why the margin depends on the query, and Section 4.5 compares Eq. (${EN.swap}) with measurements.`),

  H2("3.2", "Product quantization"),
  P(R`PQ (Jégou et al., 2011) splits each vector into $M$ sub-vectors $x=(x^{(1)},…,x^{(M)})$ with $x^{(m)}∈ℝ^{d/M}$. In each sub-space $m$ it runs k-means on the database to get a codebook $C_{m}=\c{c_{m,1},…,c_{m,K}}$ of $K=256$ centroids:`),
  EQ(R`\lim{min}{C_{m}} \sum_{x∈X}{\n{x^{(m)}−c_{m,i_{m}(x)}}^{2}}, i_{m}(x)=\lim{argmin}{j} \n{x^{(m)}−c_{m,j}}`, "kmeans"),
  P(R`Each vector is stored as its $M$ centroid indices $(i_{1}(x),…,i_{M}(x))$, one byte each, and is reconstructed as $\hat{x}=(c_{1,i_{1}(x)},…,c_{M,i_{M}(x)})$. With $M=64$ a 2,048-byte vector becomes a 64-byte code (32 times smaller), and with $M=8$ it is 256 times smaller. Splitting the vector is what makes this practical: the $M$ codebooks together describe $K^{M}=2^{8M}$ possible reconstructions while only $M·K$ centroids are stored.`),
  P(R`At query time we use *asymmetric* distance computation (ADC), as in the course. Only the database vectors are quantized and the query stays exact. For inner-product search the query first fills $M$ look-up tables $T_{m}[j]=\a{q^{(m)},c_{m,j}}$, which costs $K·d$ multiply-adds. After that, each database score is a sum of $M$ table entries:`),
  EQ(R`\hat{s}(q,x)=\a{q,\hat{x}}=\sum_{m=1}^{M}{T_{m}\b{i_{m}(x)}}, s(q,x)−\hat{s}(q,x)=\a{q,x−\hat{x}}`, "adc"),
  P(R`The second part of Eq. (${EN.adc}) matters for the rest of the report. k-means minimises the *reconstruction* error $\n{x−\hat{x}}^{2}$, but what changes the ranking is the *score* error $\a{q,x−\hat{x}}$, and that depends on the query. If queries have second-moment matrix $S=E[qq^{⊤}]$, the average squared score error is`),
  EQ(R`E_{q}\b{\a{q,x−\hat{x}}^{2}}=(x−\hat{x})^{⊤}S(x−\hat{x})`, "scoreerr"),
  P(R`This equals the k-means objective only when $S∝I$. CLIP queries are far from isotropic, and image and text queries have different $S$, so codebooks fitted to the images are not the best choice for captions. Section 3.7 uses this. We also test **OPQ** (Ge et al., 2014), which learns an orthogonal rotation $R$ before quantization so that variance is spread evenly over the $M$ sub-spaces. It does not change inner products because $\a{Rq,Rx}=\a{q,x}$.`),

  H2("3.3", "Hierarchical navigable small world graphs (HNSW)"),
  P(R`HNSW (Malkov & Yashunin, 2020) is a proximity graph with several layers. Each vector gets a maximum layer $ℓ=\floor{−ln(U)·m_{L}}$ with $U∼Uniform(0,1)$ and $m_{L}=1/ln M$, so each layer holds about $1/M$ of the vectors in the layer below. When a vector is inserted, it is linked on every layer it belongs to with up to $M$ nearby vectors (up to $2M$ on the bottom layer, layer 0). These neighbours are found with a beam search of width $efConstruction$ and filtered with a heuristic that prefers neighbours in different directions.`),
  P(R`A query starts at the single node on the top layer and moves down greedily. On each layer it keeps moving to the neighbour $v$ with the highest $s(q,v)$ until no neighbour is better than the current node. On layer 0 it runs a best-first beam search that keeps the $ef$ best candidates seen so far, and stops when the best candidate not yet expanded is worse than the worst one kept. The upper layers act as long-range shortcuts, so search cost grows roughly logarithmically with $n$, and $ef$ trades speed for recall at query time without rebuilding the graph. We use $M=32$ and $efConstruction=200$ unless stated otherwise. Each vector stores its full 2,048-byte embedding and $2M$ four-byte layer-0 links, giving $4d+8M=2,304$ bytes, plus a small overhead for the upper layers (we measured 2,320 bytes).`),

  H2("3.4", "Exact search"),
  P("Brute-force inner-product search (FAISS IndexFlatIP) gives the ground truth, i.e. the true top-100 for every query, and is our speed baseline. It needs no training and has recall 1 by definition."),
);

// ---- 3.5-3.8 failure model and improvements
add(
  P("The three methods work differently, but each decides what to return using approximate or partial information about the similarities. The rest of this section describes when that goes wrong, which properties of a query matter, and how we change each method to handle text queries better."),

  H2("3.5", "When does an approximate index lose a neighbour?"),
  P(R`Let $s_{(r)}(q)$ be the similarity between the query and its $r$-th true neighbour. LSH and PQ replace $s(q,x)$ with an estimate $\hat{s}(q,x)=s(q,x)+ε(q,x)$ and rank the whole database by it. A true neighbour $u∈N_{k}(q)$ is pushed out by a non-neighbour $v$ when`),
  EQ(R`ε(q,v)−ε(q,u)>s(q,u)−s(q,v)`, "displace"),
  P(R`The right-hand side depends only on the *query*. It is small when the true top-$k$ is not well separated from the vectors just below it. We summarise this with the **neighbour contrast** $γ(q)=s_{(1)}(q)−s_{(100)}(q)$. If the estimation error has size $σ_{ε}(q)$, what matters for recall is the ratio`),
  EQ(R`ρ(q)=\frac{σ_{ε}(q)}{γ(q)}`, "rho"),
  P(R`so halving the contrast has the same effect as doubling the error. Graph search is different. It computes exact scores, but only for the nodes it visits along its path. A greedy step from node $u$ helps only if one of $u$'s neighbours is clearly more similar to $q$ than $u$ is. When the query is far from all database vectors, every similarity is small and they are all close to each other, so the search gets little information about which way to go and its stopping rule can trigger in the wrong region. For graph search, the property that matters is how close the query is to the data, which we measure by $s_{(1)}(q)$. VIBE gives a similar qualitative explanation for the slowdown it observes on OOD queries.`),
  P("This gives two predictions that we can test. (P1) LSH and PQ, which rank by estimated scores, should lose recall on queries with low contrast $γ(q)$, whether or not the method was fitted to the data. (P2) HNSW should lose recall on queries with low $s_{(1)}(q)$, and more so on larger corpora where the search path is longer. Section 4.2 shows that text queries are worse on both measures, and Section 4.5 checks both predictions query by query. The model also suggests what to change: for PQ, reduce the score error for the queries that will actually be served; for LSH, place the hyperplanes where the queries are; and for HNSW, add edges that are useful for text queries. Each change below uses a sample of training captions that does not overlap with the test queries, and none of them changes the exact ground truth."),

  H2("3.6", "Modality-aware centring for LSH"),
  P(R`CLIP image vectors lie in a narrow cone around $μ_{I}$. For a random hyperplane, $w^{⊤}x=w^{⊤}μ_{I}+w^{⊤}(x−μ_{I})$, and when $\abs{w^{⊤}μ_{I}}$ is larger than the spread of the second term, every image lands on the same side and the bit is useless. There are two common data-dependent fixes that move the hyperplanes into the cone. One subtracts the image mean, $h_{i}(x)=1[w_{i}^{⊤}(x−μ_{I})>0]$, and the other thresholds each projection at its median over the database. Both make every bit split the images roughly in half. Projected onto a random direction, the images have a nearly symmetric distribution, so its mean and median almost coincide and the two fixes give almost the same codes (on COCO, 99% of bits agree).`),
  P(R`For a text query, though, $w^{⊤}(q−μ_{I})$ is dominated by the constant $w^{⊤}(μ_{T}−μ_{I})$. After centring on the image mean, a text query lands on the same side of most hyperplanes whatever its content. We therefore centre each modality on its own mean:`),
  EQ(R`h_{i}(x)=1\b{w_{i}^{⊤}(x−μ_{I})>0} (database), h_{i}(q)=1\b{w_{i}^{⊤}(q−μ_{T})>0} (text queries)`, "lshqa"),
  P(R`where $μ_{T}$ is estimated from the training captions. This lines up the centres of the two cones before hashing. The change targets text queries only: the database and image queries are hashed exactly as in image-mean centring, so image-query results are unchanged by construction and there is no extra build cost.`),

  H2("3.7", "Query-weighted product quantization"),
  P(R`Eq. (${EN.scoreerr}) shows that the error that matters for a given query distribution is the reconstruction error weighted by $S$. Let $A=S^{1/2}$ be the symmetric square root of the query second-moment matrix. If we index $x'=Ax$ and search with $q'=A^{−1}q$, every inner product stays exactly the same:`),
  EQ(R`\a{q',x'}=q^{⊤}A^{−1}Ax=\a{q,x}`, "qa1"),
  P("so the ground truth does not change, and ordinary k-means in the new space now minimises the expected squared score error for queries drawn from $S$:"),
  EQ(R`\n{x'−\hat{x}'}^{2}=(x−\hat{x})^{⊤}S(x−\hat{x})=E_{q}\b{\a{q,x−\hat{x}}^{2}}`, "qa2"),
  P(R`We estimate $S$ either from 200,000 training captions (*text-weighted*) or as the average $\frac{1}{2}(S_{T}+S_{I})$ of the caption and image second moments (*mixed*). We add $0.01·\bar{λ}$ to its eigenvalues so that $A^{−1}$ is well conditioned. The transform puts most of the variance into a few directions, which a fixed split into sub-vectors handles badly, so we combine it with the OPQ rotation. The extra cost at query time is one $512×512$ matrix–vector product. This is related to ScaNN's anisotropic loss (Guo et al., 2020), since both weight quantization error by how much it changes the inner products instead of by its Euclidean size, but ours is a closed-form transform based on the query statistics.`),

  H2("3.8", "Caption-guided edges for HNSW"),
  P(R`HNSW links images that are close *to each other*. The true neighbours of a caption, however, can be images that are not close to each other at all and only share their relevance to the caption. For example, a street scene and a close-up of a bus can both match “a red bus in a city”. A greedy search starting far away may then have no short path between them. Following RoarGraph and OOD-DiskANN, we add edges based on what the queries consider close. For each training caption we compute its exact top-$K$ images ($K=16$) and link its nearest image to the other $K−1$ in both directions (Algorithm 1).`),
  ...algorithm("**Algorithm 1.** Caption-guided edge insertion", [
    [0, "**Input:** HNSW graph $G$ with layer-0 capacity $2M$ per node; image vectors $X$; training captions $T$; $K=16$; per-node cap $E_{max}=16$"],
    [0, "**for each** caption $t∈T$ **do**"],
    [1, "$(u_{1},…,u_{K})←$ exact top-$K$ images of $t$ in $X$"],
    [1, "propose pairs $(u_{1},u_{j})$ and $(u_{j},u_{1})$ for $j=2,…,K$"],
    [0, "count the support $c(u,v)$ of each proposed pair; discard pairs already in $G$"],
    [0, "**for each** node $u$ **do**"],
    [1, "append $u$'s proposed neighbours to its layer-0 list in decreasing $c(u,v)$, while free slots remain and at most $E_{max}$ are added"],
    [0, "**Output:** $G'$, with the same memory and the same search procedure"],
  ]),
  P(R`The new edges go into layer-0 slots that HNSW has already allocated but left empty. The average layer-0 degree is only 21.6 on COCO and 23.1 on LAION out of $2M=64$, so the index size does not change, and the search algorithm is not modified. The only extra cost is the offline exact top-$K$ search for each training caption. To see *why* the edges help, we also test two other caption-based changes as ablations (Section 4.7). The first starts the layer-0 search from the true neighbours of the most similar training captions, which tests whether the entry point is the problem. The second inserts the captions themselves into the graph as nodes that are never returned, which tests whether the graph lacks landmarks near text queries.`),
);

// ---- 4 experiments
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
  P("**Datasets.** The two datasets differ in size (8×) and in the kind of text, clean human-written captions on COCO and noisy web text on LAION (Table 2). Using both lets us check that the results do not depend on one benchmark. On COCO the image and text queries describe the same 1,000 held-out images, so the only difference between the two query sets is which encoder produced them."),
  P("**Protocol.** We follow the VIBE protocol (Jääsaari et al., 2026). (i) *Build on the corpus only*: codebooks, hyperplane statistics and graphs never see a test query. The improved methods in Sections 3.6 to 3.8 may use the training captions but never the test queries. (ii) *One index for both query types*: each index is built once and searched with image and text queries using the same parameters, so any difference comes from the queries. (iii) *Timing*: queries are sent one at a time on a single CPU thread pinned to one core, and we report the fastest of several runs. All runs used the same server (2× AMD EPYC 7742) with FAISS 1.15.1, and all random seeds are fixed."),
  P(R`**Metrics.** Our main metric is recall@100 (Eq. ${EN.recall}). We also report recall@10, single-thread queries per second (QPS), index size in bytes per vector, and build time. The gap between the two query types is`),
  EQ(R`Δ=Recall@100(Q_{image})−Recall@100(Q_{text})`, "delta"),
  P("measured on the same index with the same setting. We never average image and text recall, because the difference between them is what we are studying."),

  H2("4.2", "How text and image queries differ"),
  P("We first check the modality gap in our data. Table 3 shows that the mean text query points in a very different direction from the mean image (cosine 0.31 to 0.36), while the mean held-out image query matches it almost exactly (0.999)."),
  ...table(3, "How the queries relate to the indexed images. $γ$ is the neighbour contrast from Section 3.5.",
    ["Dataset", "Query type", "$cos$(query mean, image mean)", "$s_{(1)}$", "$s_{(100)}$", "Contrast $γ$"],
    [["COCO", "image", "0.999", "0.858", "0.771", "0.087"], ["COCO", "text", "0.309", "0.345", "0.299", "0.046"],
     ["LAION", "image", "0.999", "0.825", "0.733", "0.092"], ["LAION", "text", "0.357", "0.355", "0.300", "0.055"]],
    [1.2, 1.2, 2.4, 1.3, 1.3, 1.5]),
  ...figure("fig1_neighbour_similarity.png", 1720, 681, 1, "Similarity of each query to its 1st to 100th true neighbour (solid: image queries; dashed: text queries; band: 10th to 90th percentile)."),
  P("Figure 1 shows that text queries differ in the two ways described in Section 3.5. First, they are far from the data: the most similar image has a similarity of 0.35, compared with 0.86 for an image query. Second, their neighbours are bunched together: the contrast $γ$ between the 1st and 100th neighbour is about half that of image queries (0.046 vs 0.087 on COCO). By Eq. (" + EN.rho + "), the same scoring error therefore does about twice as much damage for a text query."),

  H2("4.3", "Speed, accuracy and memory"),
  P("With image queries, each method behaves as its design suggests (Figures 2 and 3, Table 4)."),
  ...figure("fig2_recall_vs_qps.png", 1722, 854, 2, "Recall@100 vs single-thread queries per second. Colour shows the method; solid lines and filled markers are image queries, dashed lines and hollow markers are text queries; the star is exact search."),
  ...figure("fig3_recall_vs_memory.png", 1685, 854, 3, "Recall@100 vs index size in bytes per vector."),
  ...table(4, "Best setting of each method with image queries.",
    ["Dataset", "Method", "Setting", "Recall@100", "QPS", "Speed-up", "Bytes / vector", "Build (s)"],
    [["COCO", "HNSW", "$ef$ = 128", "0.992", "2,235", "30×", "2,320", "26"],
     ["COCO", "PQ", "$M$ = 64", "0.657", "181", "2.5×", "68", "5"],
     ["COCO", "LSH", "$b$ = 4,096", "0.837", "128", "1.7×", "581", "13"],
     ["COCO", "Exact", "-", "1", "74", "1×", "2,048", "0.2"],
     ["LAION", "HNSW", "$ef$ = 512", "0.984", "488", "51×", "2,320", "226"],
     ["LAION", "PQ", "$M$ = 64", "0.560", "22", "2.3×", "65", "12"],
     ["LAION", "LSH", "$b$ = 4,096", "0.805", "17", "1.8×", "520", "72"],
     ["LAION", "Exact", "-", "1", "9.6", "1×", "2,048", "1.3"]],
    [1.1, 1, 1.4, 1.2, 1, 1.1, 1.3, 1]),
  P("HNSW is the fastest and most accurate, with recall@100 of at least 0.98 at 30 to 51 times the speed of exact search. Its advantage grows with corpus size because it computes only 1,400 to 2,000 similarities per query, while a scan computes $n$. The price is memory (13% *more* than the raw vectors, because of the links) and build time (226 s for 1M vectors on 32 threads). PQ is the opposite. It stores 8 to 64 bytes per vector instead of 2,048 (32 to 240 times smaller), but on its own it is neither accurate (recall@100 at most 0.66) nor fast, since ADC still visits every code, so in practice it is followed by re-ranking (Section 4.7). LSH needs no training and builds in seconds, but it needs 4,096 bits (512 bytes) to reach recall 0.80 to 0.84, and at that size its full scan is barely faster than exact search."),
  P("Table 5 compares the two compression methods at the same code length. PQ's learned codebooks beat LSH's random hyperplanes at every size. The difference is small for image queries but 2.8 to 30 times for text queries. LSH codes, on the other hand, are 5 to 10 times faster to scan, because a popcount is cheaper than $M$ table look-ups. This is the data-dependent versus data-independent trade-off from the course, and the text columns already suggest that not learning from the data does not protect LSH from the modality gap."),
  ...table(5, "Recall@100 (image / text) at equal bits per vector; PQ uses $M=b/8$.",
    ["Bits / vector", "COCO LSH", "COCO PQ", "LAION LSH", "LAION PQ"],
    [["64", "0.155 / 0.011", "0.303 / 0.179", "0.075 / 0.002", "0.111 / 0.059"],
     ["128", "0.270 / 0.025", "0.392 / 0.229", "0.187 / 0.008", "0.198 / 0.114"],
     ["256", "0.420 / 0.071", "0.505 / 0.336", "0.337 / 0.030", "0.345 / 0.232"],
     ["512", "0.577 / 0.181", "0.657 / 0.515", "0.504 / 0.113", "0.560 / 0.438"]],
    [1.2, 2, 2, 2, 2]),

  H2("4.4", "The cost of querying with text"),
  P("Every method loses recall on text queries at every setting (Figure 4, Table 6), and all the gaps are many standard errors above zero."),
  ...figure("fig4_param_sweep_gap.png", 2498, 1222, 4, "Image (solid) and text (dashed) recall@100 across each method's parameter. The shaded area is the gap Δ."),
  ...table(6, "The gap Δ = recall@100(image) − recall@100(text) at both ends of each parameter sweep. The last column is the largest Δ at any setting in the sweep.",
    ["Dataset", "Method", "Smallest setting", "Largest setting", "Largest Δ"],
    [["COCO", "HNSW", "$ef$ 100: 0.986 vs 0.884 (Δ 0.101)", "$ef$ 1024: 1.000 vs 0.996 (Δ 0.004)", "0.101"],
     ["COCO", "PQ", "$M$ 8: 0.303 vs 0.179 (Δ 0.123)", "$M$ 64: 0.657 vs 0.515 (Δ 0.142)", "0.169"],
     ["COCO", "LSH", "$b$ 64: 0.155 vs 0.011 (Δ 0.144)", "$b$ 4096: 0.837 vs 0.559 (Δ 0.279)", "0.395"],
     ["LAION", "HNSW", "$ef$ 100: 0.939 vs 0.749 (Δ 0.190)", "$ef$ 1024: 0.988 vs 0.956 (Δ 0.031)", "0.190"],
     ["LAION", "PQ", "$M$ 8: 0.111 vs 0.059 (Δ 0.053)", "$M$ 64: 0.560 vs 0.438 (Δ 0.122)", "0.122"],
     ["LAION", "LSH", "$b$ 64: 0.075 vs 0.002 (Δ 0.072)", "$b$ 4096: 0.805 vs 0.515 (Δ 0.290)", "0.401"]],
    [1.1, 1, 2.7, 2.7, 1.1]),
  P("How the gap changes with the parameter differs between the methods. For HNSW a wider search closes most of it: Δ falls from 0.10 to 0.004 on COCO and from 0.19 to 0.03 on LAION as $ef$ increases. For PQ, longer codes do not help, and Δ stays between 0.12 and 0.17 on COCO. LSH has the largest gap, up to 0.40, and at 512 bits text queries get only 22 to 31% of the recall that image queries get."),
  P("The LSH result is the opposite of what we expected. LSH learns nothing from the images, so its loss cannot come from an index fitted to the wrong data. It has to come from the text queries themselves, which is what Section 4.5 examines."),
  P("To put the gap in terms of cost, Table 7 shows how much more search effort HNSW needs for text queries to reach the recall that image queries get at a cheap setting."),
  ...table(7, "Extra search effort HNSW needs for text queries to match image-query recall.",
    ["Dataset", "Image queries: $ef$ → recall (QPS)", "Text queries need $ef$", "Text QPS", "Slow-down"],
    [["COCO", "100 → 0.986 (2,700)", "≈ 490", "≈ 790", "3.4×"],
     ["COCO", "128 → 0.992 (2,235)", "≈ 700", "≈ 580", "3.9×"],
     ["LAION", "100 → 0.939 (1,870)", "≈ 650", "≈ 400", "4.7×"],
     ["LAION", "128 → 0.952 (1,553)", "≈ 900", "≈ 295", "5.3×"]],
    [1.1, 3, 1.8, 1.4, 1.4]),
  P("A search engine tuned on image queries and then used with captions would get 10 to 20 points less recall than its benchmark showed, and getting that recall back would make search 3 to 5 times slower."),

  H2("4.5", "Checking the model"),
  P("We now test the two predictions from Section 3.5 on individual queries (Figure 5)."),
  ...figure("fig5_which_queries_fail.png", 1982, 1262, 5, "Per-query recall@100, with queries grouped into 12 equal-size bins. Left: HNSW against the query's similarity to its nearest image, $s_{(1)}$. Right: PQ against the neighbour contrast $γ$."),
  P(R`**(P1) Low contrast hurts LSH and PQ.** For PQ, per-query recall increases with neighbour contrast (Spearman correlation up to +0.51), and 39% of COCO text queries get recall below 0.5, compared with 3% of image queries. The absolute PQ score error is actually *smaller* for text queries (RMS 0.038 vs 0.052 on COCO at $M=64$). The difference is in the margin, so the error-to-contrast ratio of Eq. (${EN.rho}) is 0.78 for text and 0.57 for images.`),
  P(R`For LSH we can check the prediction with numbers. We first confirmed that the measured per-bit collision rate matches Charikar's $1−θ/π$ to within 0.001 for *both* query types, so the hash functions treat a text pair and an image pair at the same angle in the same way. The difference is entirely in the margins. On LAION, the median angular margin between a query's 100th and 1,000th neighbours is 4.7° for image queries but only 1.8° for text queries. Table 8 compares the swap probability from Eq. (${EN.swap}) with the rate measured from the actual codes.`),
  ...table(8, "LSH on LAION: probability that the 1,000th neighbour is ranked above the 100th, predicted by the binomial model (Eq. " + EN.swap + ") and measured (seed 0).",
    ["Query type", "Margin", "$b$", "$z$", "Predicted", "Measured", "Recall@100"],
    [["image", "4.7°", "256", "0.82", "0.199", "0.169", "0.337"],
     ["image", "4.7°", "512", "1.16", "0.126", "0.104", "0.504"],
     ["image", "4.7°", "4,096", "3.28", "0.006", "0.007", "0.805"],
     ["text", "1.8°", "256", "0.30", "0.380", "0.349", "0.030"],
     ["text", "1.8°", "512", "0.42", "0.333", "0.333", "0.113"],
     ["text", "1.8°", "4,096", "1.19", "0.123", "0.091", "0.517"]],
    [1.2, 1, 1, 1, 1.2, 1.2, 1.3], "The model slightly overestimates the swap rate because the rows of a random rotation are orthogonal, not independent."),
  P(R`The predictions are close to the measurements. The model also explains why the LSH gap is so large. Since $z∝δ\sqrt{b}$, a margin that is 2.6 times smaller needs $2.6^{2}≈7$ times more bits for the same separation. Text queries at 4,096 bits ($z=1.19$, recall 0.517) behave almost exactly like image queries at 512 bits ($z=1.16$, recall 0.504). That is eight times the code length, or three doublings, which matches the horizontal shift between the two LSH curves in Figure 4. LSH suffers the most because it cannot adapt to the data: unlike PQ, nothing in how it is built can put more precision where the text queries' margins are small.`),
  P(R`**(P2) Distance from the data hurts HNSW.** HNSW's per-query recall increases with the query's similarity to its nearest image (Spearman correlation +0.55 on COCO and +0.64 on LAION; Figure 5, left), and text queries sit at the low end of this range. On LAION at $ef=100$, text queries even compute 1% *fewer* similarities than image queries, yet lose 19 points of recall (Figure 6). The beam search decides it has finished while it is still in the wrong region, which is what the argument in Section 3.5 predicts.`),
  ...figure("fig6_hnsw_extra_work.png", 990, 638, 6, "Extra similarity computations per text query compared with an image query (HNSW).", 3.6),

  H2("4.6", "Ablation studies"),
  H3("HNSW build parameters"),
  P("Figure 7 varies HNSW's build parameters on COCO. More links per node reduce the gap. Going from $M=8$ to $M=64$ raises text recall at $ef=100$ from 0.79 to 0.89 and lowers Δ from 0.155 to 0.098, at the cost of 21% more memory, and most of the improvement is already there at $M=16$ to 32. $efConstruction$ has no consistent effect: text recall is 0.92, 0.86, 0.88 and 0.90 for 40, 100, 200 and 400. Repeated builds with the same seed give identical results, so these differences are real, but they follow no trend. A more thorough build makes the graph better at connecting similar images, which is not what text queries need."),
  ...figure("fig7_hnsw_build_params.png", 1701, 674, 7, "HNSW recall@100 against graph degree $M$ (left) and $efConstruction$ (right), COCO."),
  H3("Corpus size"),
  P("To separate the effect of size from the effect of the dataset, Figure 8 uses random LAION subsets, with the ground truth recomputed for each one. HNSW's gap grows steadily with $n$. At $ef=100$ it is 0.10 at 10k images, 0.13 at 100k and 0.19 at 1M, which fits prediction (P2): a longer greedy path has more chances to stop in the wrong region. PQ's gap does not grow (0.10 to 0.12 at every size). So HNSW's problem with text queries gets worse with scale, while PQ's stays fixed."),
  ...figure("fig8_scaling.png", 1701, 717, 8, "Recall@100 against corpus size (random LAION subsets). Left: HNSW; right: PQ."),
  H3("Removing the offset between modalities"),
  P(R`If the loss happens because text and images are in different regions, removing the offset between them should reduce it. As a test, we subtract $μ_{I}$ from the images and $μ_{T}$ from the text queries, normalise again, and rebuild every index *and the ground truth* in this centred space (Table 9).`),
  ...table(9, "Effect of centring each modality on its own mean (recall@100 image / text, and Δ).",
    ["Dataset", "Method", "Original space", "Centred space"],
    [["COCO", "HNSW, $ef$ = 100", "0.986 / 0.884 (Δ 0.101)", "0.987 / 0.931 (Δ 0.057)"],
     ["COCO", "PQ, $M$ = 64", "0.656 / 0.512 (Δ 0.143)", "0.738 / 0.573 (Δ 0.165)"],
     ["LAION", "HNSW, $ef$ = 100", "0.939 / 0.749 (Δ 0.190)", "0.964 / 0.868 (Δ 0.096)"],
     ["LAION", "PQ, $M$ = 64", "0.561 / 0.437 (Δ 0.124)", "0.678 / 0.514 (Δ 0.164)"]],
    [1.1, 2, 3, 3]),
  P("Centring cuts HNSW's gap roughly in half, which supports (P2). For PQ it roughly doubles the contrast of *both* query types (COCO: 0.087 to 0.168 for images and 0.046 to 0.090 for text), so both recalls go up. Text queries still have half the contrast of image queries, though, and the gap stays, which supports (P1). Centring cannot be used as an improvement in practice, because it changes which images count as nearest: only 62 to 72% of each text query's original top-100 is still in its new top-100. The improvements in the next section keep the original ground truth."),

  H2("4.7", "Results of the improved methods"),
  P("Everything in this section is measured against the original exact top-100, so the numbers can be compared directly with Sections 4.3 to 4.5."),
  H3("HNSW with caption-guided edges"),
  ...table(10, "HNSW variants at $ef$ = 100 (recall@100). Variants 1 and 2 are the ablations described in Section 3.8. Bold: best recall per dataset; underlined: second best (COCO only, since LAION has two rows).",
    ["Dataset", "Variant", "Image", "Text", "Δ", "Extra memory"],
    [["COCO", "Baseline", "__0.986__", "__0.884__", "0.101", "-"],
     ["COCO", "1. Caption entry points", "__0.986__", "0.877", "0.108", "+2,328 B/vector"],
     ["COCO", "2. Captions as graph nodes", "0.985", "0.078", "0.907", "+232 B/vector"],
     ["COCO", "3. Caption-guided edges (Alg. 1)", "**0.987**", "**0.947**", "0.040", "none"],
     ["LAION", "Baseline", "0.939", "0.749", "0.190", "-"],
     ["LAION", "3. Caption-guided edges (Alg. 1)", "**0.950**", "**0.818**", "0.132", "none"]],
    [1.1, 3.2, 1, 1, 1, 1.7]),
  ...figure("fig9_hnsw_fixes.png", 1741, 879, 9, "HNSW variants: recall@100 against QPS (dashed: text queries). Purple: caption-guided edges (lighter shades use fewer training captions); green: caption entry points."),
  P("Table 10 and Figure 9 compare the three caption-based changes. Starting the search from better entry points (variant 1) does not help, so the descent through the upper layers is not where the search goes wrong. Inserting the captions themselves as nodes (variant 2) makes things much worse. A caption node is far more similar to a text query (cosine about 0.8) than any image is (about 0.3), so caption nodes fill the whole beam and the search never reaches the images. This is also why RoarGraph uses training queries only to choose edges between data points and leaves them out of its final graph."),
  P("Caption-guided edges (variant 3) work. On COCO, text recall goes from 0.88 to 0.95, Δ drops by 60%, and image recall goes up slightly. The gain also holds at equal speed: plain HNSW reaches text recall 0.948 at about 1,470 QPS, while the modified graph reaches 0.947 at about 2,170 QPS, 1.5 times faster. On LAION the benefit depends on how many training captions are used. With 10% of them there is no gain at equal speed, and with all of them text recall improves by 7 points at $ef=100$ and by 3 to 6 points at equal speed."),
  P("Building the edges needs an offline exact top-16 search for each training caption, which took 73 s for COCO and 1.7 h on 32 cores for LAION's 1M captions. It adds 6.6 edges per node on COCO and 11.0 on LAION on average. Together these results support the model: the search starts in a reasonable place, but the image graph is missing links between images that are relevant to the same caption without being close to each other."),

  H3("Query-weighted PQ"),
  P("Table 11 combines the two parts of Section 3.7, query weighting and the OPQ rotation, and also includes the mixed weighting. The last two columns give the error-to-contrast ratio $ρ$ from Eq. (" + EN.rho + ") at $M$ = 64, computed from the RMS score error over each query's true top-100."),
  ...table(11, "PQ variants at $M$ = 64 bytes per vector: recall@100 (image / text, Δ) and error-to-contrast ratio $ρ$ (image / text). Bold: best in each column for each query type (highest recall, lowest $ρ$); underlined: second best.",
    ["Variant", "Weights", "OPQ", "COCO recall", "LAION recall", "COCO $ρ$", "LAION $ρ$"],
    [["PQ (baseline)", "none", "no", "0.656 / 0.512 (Δ 0.143)", "0.561 / 0.437 (Δ 0.124)", "0.57 / 0.78", "0.72 / 0.82"],
     ["OPQ", "none", "yes", "__0.773__ / 0.665 (Δ 0.108)", "__0.662__ / __0.537__ (Δ 0.124)", "**0.26** / 0.36", "**0.47** / __0.56__"],
     ["Text-weighted PQ", "captions", "no", "0.160 / 0.461", "0.124 / 0.392", "0.99 / 0.82", "0.95 / 0.90"],
     ["Text-weighted OPQ", "captions", "yes", "0.632 / **0.793** (Δ −0.161)", "0.554 / **0.579** (Δ −0.025)", "__0.43__ / **0.21**", "0.65 / **0.53**"],
     ["Mixed OPQ", "50/50", "yes", "**0.809** / __0.732__ (Δ 0.077)", "**0.691** / 0.519 (Δ 0.171)", "**0.26** / __0.26__", "__0.50__ / 0.60"]],
    [1.6, 1.0, 0.75, 2.35, 2.35, 1.1, 1.1]),
  ...figure("fig10_pq_fixes.png", 1983, 1302, 10, "PQ variants: recall@100 against code size, for image queries (top) and text queries (bottom)."),
  P("The OPQ rotation is necessary. Weighting on its own puts most of the variance into a few directions, which a fixed split into sub-vectors handles badly, so image recall drops to 0.12 to 0.16 and $ρ$ gets close to 1. With the rotation, text weighting helps text queries a lot. It raises COCO text recall from 0.51 to 0.79 (+55%) at the same 64 bytes, to the point where text queries do better than image queries, and on LAION it removes the gap at every code size. If both query types matter, mixed weighting is the better choice: on COCO it gives the best image recall and the second-best text recall, and halves Δ."),
  P("The ratio $ρ$ ranks the variants in almost the same order as recall on COCO, and mixed weighting makes it equal for the two query types (0.26 and 0.26), which is where the gap is smallest. On LAION the ranking holds for text queries and mostly holds for image queries. This agrees with Section 3.5: what raises recall is lowering the score error relative to the contrast, not lowering the reconstruction error."),
  P("**Re-ranking.** Another common option is to re-score PQ's top candidates with the full vectors. Re-ranking the top 1,000 (at most 21% slower) raises recall at $M$ = 64 to 0.998 / 0.974 on COCO and 0.968 / 0.895 on LAION. However, this needs the full vectors in RAM or on SSD, which removes PQ's memory advantage, and it cannot recover neighbours that never made it into the candidate list. With short codes the gap even grows as the list gets longer (LAION, $M$ = 16: Δ goes from 0.12 to 0.20 between 200 and 1,000 candidates). The two approaches work well together: better codes put more true neighbours into the list, and re-ranking then puts them in the right order."),

  H3("Modality-aware LSH"),
  ...figure("fig11_lsh_variants.png", 1983, 682, 11, "LSH variants: text-query recall@100 against code length. Median thresholds and image-mean centring overlap."),
  P("Modality-aware centring is meant to improve text queries, not image queries, so Figure 11 compares plain LSH with the three variants from Section 3.6 on text queries only. For image queries the results are the same as with image-mean centring, since image queries are hashed in exactly the same way."),
  P(R`The two image-based adjustments, median thresholds and image-mean centring, make text recall much worse. At 4,096 bits it drops from 0.56 to 0.22 on COCO and from 0.52 to 0.07 on LAION. Every hyperplane now passes through the image cone, so a text query, which lies outside it, ends up on the same side of most hyperplanes, and its code hardly changes from one query to the next, as predicted in Section 3.6. Modality-aware centring (Eq. ${EN.lshqa}) fixes this and does better than plain LSH. Text recall roughly doubles at medium code lengths (COCO, 512 bits: 0.18 to 0.33; LAION, 1,024 bits: 0.23 to 0.40), and the LAION gap at 4,096 bits falls from 0.29 to 0.12.`),

  H2("4.8", "Success and failure cases"),
  P("Figure 12 shows three COCO test captions, picked to show the two kinds of failure, together with the exact top-10 images and each method's top-10."),
  ...figure("fig12_examples.png", 1722, 1257, 12, "Three COCO test captions. Rows: exact top-10, HNSW ($ef$ = 100), HNSW with caption-guided edges, PQ ($M$ = 64). Green border: in the exact top-10; red: not."),
  P("For the long caption about a train, HNSW gets stuck in a nearby region and drifts towards lorries and car parks (recall@10 = 0.5), while the graph with caption-guided edges returns exactly the true top-10. For the baseball caption, PQ finds the right topic but the wrong order. Every PQ result is a reasonable baseball photo, but none of them is in the exact top-10, because hundreds of images have almost the same score and small errors reshuffle them. This also shows a limitation of measuring recall against exact search, since a user might be happy with PQ's results. These cases were picked to show failures. Over all captions HNSW usually succeeds, with text recall@10 of 0.949 for plain HNSW and 0.987 with caption-guided edges."),
);

// ---- 5 discussion
add(H1("5", "Discussion"),
  H2("5.1", "Strengths and weaknesses"),
  ...table(12, "Summary of the three methods, based on our measurements.",
    ["Method", "Strengths", "Weaknesses", "Best use"],
    [["HNSW", "Recall@100 of 0.98 to 0.99 at 30 to 51× exact speed; tunable at query time; text loss shrinks with $ef$ and caption-guided edges", "More memory than the raw vectors; slow to build; text loss grows with corpus size", "Low-latency search when there is enough RAM"],
     ["PQ", "32 to 240× smaller; fast to build; query-weighted OPQ recovers text recall", "Recall at most 0.66 on its own; flat scan barely faster than exact; gap does not shrink with longer codes", "Memory-limited systems, as a first stage before re-ranking"],
     ["LSH", "No training; builds in seconds; scans 5 to 10× faster than PQ at equal bits; simple error model", "Needs 4,096 bits for recall 0.8; largest text gap (up to 0.40); image-based variants destroy text recall", "Streaming data and quick baselines, with modality-aware centring"]],
    [1, 3.2, 3.2, 2.2]),
  H2("5.2", "Key factors affecting performance"),
  P("Our experiments point to two separate causes of the text-query loss. The first is low neighbour contrast. It is a property of the queries, which is why it shows up even in LSH, and longer codes only reduce its effect slowly ($z∝δ\\sqrt{b}$). What helps is putting the precision of the code where the queries need it, with query-weighted codebooks or hash functions centred on each modality. The second is distance from the data, which only affects graph search. It grows with corpus size, is roughly halved by centring, and is reduced by adding edges between images that are relevant to the same caption, while a better starting point makes no difference. In both cases, building the index with the expected queries in mind, and not only the stored data, gave the biggest improvement."),
  P("The parameters of each method behave as follows. For LSH, code length $b$ is the only option, and its benefit shrinks as $1/\\sqrt{b}$. For PQ, $M$ sets the trade-off between memory and accuracy, but for text queries the choice of objective (rotation and weighting) matters more than doubling $M$. For HNSW, $ef$ is the most useful setting at query time, $M$ helps up to about 32, and $efConstruction$ has no reliable effect on text recall."),
  H2("5.3", "Practical recommendations"),
  B("**Tune and evaluate with the queries the system will actually get.** An HNSW index tuned to 0.94 to 0.99 recall on image queries gives 0.75 to 0.88 on captions, and matching the image-query recall makes search 3 to 5 times slower."),
  B("**Use a sample of real queries at build time if one is available.** Caption-guided edges cost no memory, and query-weighted OPQ costs one $512×512$ matrix."),
  B("**Do not centre or threshold using image statistics alone** when the queries come from another modality."),
  H2("5.4", "Limitations"),
  B("We only use one encoder (CLIP ViT-B/32). Newer encoders may have a smaller modality gap."),
  B("All timing is single-threaded on CPU. To finish in time, independent jobs ran at the same time on separate cores of one server. This may lower absolute QPS, mainly for the memory-heavy scans on LAION, but it does not affect recall."),
  B("Recall is measured against exact search, not against human judgement, so an approximate “miss” can still be a relevant image (Figure 12)."),
  B("The improved methods need a sample of training queries from the same distribution as the real queries. We did not test what happens when the two distributions differ."),
);

// ---- 6 conclusion
add(H1("6", "Conclusion"),
  P("We compared hashing (LSH), quantization (PQ) and graph (HNSW) indexes for text-to-image search with CLIP embeddings on COCO (122k images) and LAION (1M images), searching each index with both image and text queries. With image queries we see the usual trade-off: HNSW is the fastest and most accurate, PQ is the smallest, and LSH is the cheapest to build. With text queries every method loses recall at every setting, and LSH loses the most even though it does not learn from the data."),
  P("Two properties of text queries explain this. Their true neighbours have very similar scores, so the scoring errors of LSH and PQ reorder them. For LSH, a binomial model matches the measured swap rates and explains why text queries need about eight times longer codes. Text queries are also far from every image, which makes HNSW's greedy search stop in the wrong place, and this gets worse as the corpus grows. Based on this, we improved each method while keeping the exact ground truth: caption-guided edges in HNSW's unused slots (COCO text recall from 0.88 to 0.95 with no extra memory), query-weighted OPQ (PQ text recall from 0.51 to 0.79 at 64 bytes), and modality-aware centring for LSH (text recall roughly doubled at medium code lengths)."),
  P("Our main conclusion is that indexes for cross-modal search should be built, tuned and evaluated with the kind of queries they will actually receive. Possible future work includes computing caption-guided edges on a GPU for web-scale data, combining the graph and quantization improvements in IVF-PQ or disk-based graph indexes, and repeating the study with newer encoders."),
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
    "E. Jääsaari, V. Hyvönen, M. Ceccarello, T. Roos, M. Aumüller. VIBE: Vector Index Benchmark for Embeddings. Journal of Data-centric Machine Learning Research, 2026. arXiv:2505.17810.",
    "S. Jaiswal et al. OOD-DiskANN: efficient and scalable graph ANNS for out-of-distribution queries. arXiv:2211.12850, 2022.",
    "H. Jégou, M. Douze, C. Schmid. Product quantization for nearest neighbor search. IEEE TPAMI 33(1), 2011.",
    "W. Liang, Y. Zhang, Y. Kwon, S. Yeung, J. Zou. Mind the gap: understanding the modality gap in multi-modal contrastive representation learning. NeurIPS, 2022.",
    "T.-Y. Lin et al. Microsoft COCO: common objects in context. ECCV, 2014.",
    "Y. A. Malkov, D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI 42(4), 2020.",
    "A. Radford et al. Learning transferable visual models from natural language supervision. ICML, 2021.",
    "C. Schuhmann et al. LAION-400M: open dataset of CLIP-filtered 400 million image-text pairs. arXiv:2111.02114, 2021.",
  ].map((r) => new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(r, { size: 22 }),
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
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [small([new TextRun({ text: "SC4020 Technical Review: Cross-Modal Nearest Neighbour Search", size: 17, italics: true, color: "555555" })])] }) },
    footers: { default: new Footer({ children: [small([new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "555555" })])] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
