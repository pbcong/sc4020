// Builds the technical-review report. Usage (from report/): npm install && node build.js [out.docx]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, LevelFormat, Footer, Header, PageNumber, TableLayoutType, TabStopType,
} = require("docx");
const { math } = require("./omml");

const FIG = path.join(__dirname, "..", "results", "figures") + path.sep;
const OUT = process.argv[2] || path.join(__dirname, "group_21_report.docx");
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
    children: [new TextRun({ text: "SC4020 Project 1: Technical Review Report", bold: true, size: 34 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Topic: Similarity Search (LSH, PQ and HNSW for Text-to-Image Search)", bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Group 21", bold: true, size: 22 })] }),
  ...[["Pham Ba Cong", "U2331760J"], ["Phan The Manh", "[matric no.]"], ["Chu Gia Han", "[matric no.]"], ["Pham Gia Phuc", "[matric no.]"]]
    .map(([name, id]) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
      children: [new TextRun({ text: `${name} (${id})`, size: 22 })] })),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 320 },
    children: [new TextRun({ text: "Nanyang Technological University", size: 20, italics: true })] }),
);

// ---- abstract
const ABS = { left: 567, right: 567 };
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Abstract", bold: true, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: ABS, spacing: { after: 280, line: 252 }, children: runs(
    "Text-to-image search engines turn captions and images into vectors with CLIP and then search the image vectors with an approximate nearest neighbour (ANN) method. The index is built from images, but the queries are captions, and CLIP places text and image vectors in different regions of the space (the *modality gap*). We compare three ANN methods from three families, LSH with random projection (hashing), Product Quantization (PQ) and HNSW (graph), on MS-COCO (122k images) and LAION (1M images). Every index is searched with both image queries and text queries using the same settings. We measure recall, the fraction of the exact top-100 results that each method returns. All three methods lose recall on text queries. Surprisingly, LSH, which does not learn from the data, loses the most (up to 0.40, against 0.17 for PQ and 0.19 for HNSW). We show that two properties of text queries cause this: their top-k neighbours have very similar scores, and they are far from every image. Based on this, we improve each method using a sample of training captions. For HNSW, we add links between images that match the same caption, which raises text recall on COCO from 0.88 to 0.95 with no extra memory. For PQ, we fit the centroids to reduce the score error for captions instead of images, which raises text recall from 0.51 to 0.79. For LSH, we shift text and image vectors by their own mean before hashing, which roughly doubles text recall at medium code lengths.", { size: 22 }) }),
);

// ---- 1 introduction
add(H1("1", "Introduction"),
  P("Similarity search finds the top-$k$ nearest neighbours of a query in a database. The simplest method is linear search (exhaustive search): compute the similarity between the query and every item in the database and return the $k$ most similar ones. The result is exact, but each query needs $O(n)$ similarity calculations, which is slow for large databases. On our LAION database of one million images, one linear search takes 104 ms on a single thread. Approximate nearest neighbour (ANN) methods such as LSH and PQ return approximate results in exchange for much faster search or much less memory."),
  P("Models such as CLIP (Radford et al., 2021) map images and text into the same 512-dimensional space, so the same ANN methods can be used to find images with a caption (*cross-modal* search). However, the two modalities do not overlap in this space. Each one sits in its own narrow cone, and the two cones are far apart (Liang et al., 2022). In our data, a caption and its own image have a cosine similarity of only about 0.3. So an index built from images receives queries from a region it has never seen. The VIBE benchmark (Jääsaari et al., 2026) shows that ANN methods become much slower on such out-of-distribution (OOD) queries, and graph methods such as RoarGraph (Chen et al., 2024) and OOD-DiskANN (Jaiswal et al., 2022) use training queries to handle them. We study the problem across three different ANN families. As a baseline, we also search each index with image queries. These come from the same distribution as the database (in-distribution), as in standard ANN benchmarks. Comparing them with text queries on the same index shows how much recall is lost only because the query is text."),
  P("We pick one method from each family: **LSH** with random projection (hashing, does not learn from the data) and **Product Quantization** (quantization, learns centroids from the data), which are both covered in the lecture, and **HNSW** (graph), which is outside the course and is the default index in vector databases such as pgvector, Milvus and Weaviate. We expected PQ, whose centroids are fitted to images, to suffer most from text queries, and LSH to suffer least. The results were the opposite, and explaining why became the main thread of this report. We:"),
  B("**Compare** the three methods and linear search on two datasets, COCO (122k images) and LAION (1M images), with both image and text queries, measuring recall, speed, memory and build time."),
  B("**Explain** why text queries lose recall and check the explanation query by query. For LSH, a simple probability model predicts the loss."),
  B("**Run ablations** on HNSW's build parameters, the database size and the offset between the two modalities."),
  B("**Improve** each method for text queries using a sample of training captions, without changing the exact ground truth."),
);

// ---- 2 methods
add(H1("2", "Methods"),
  H2("2.1", "Problem and evaluation"),
  P(R`Given a database $X=\c{x_{1},…,x_{n}}⊂ℝ^{d}$ and a query $q$, $k$-nearest neighbour search returns the $k$ items $N_{k}(q)$ with the highest similarity $s(q,x)$. All our vectors are L2-normalised, so the inner product, cosine similarity and L2 distance give the same ranking, since $\n{q−x}^{2}=2−2\a{q,x}$. We use $s(q,x)=\a{q,x}$. An ANN method returns an approximate list $\tilde{N}_{k}(q)$, which we compare with the exact list from linear search:`),
  EQ(R`Recall@k=\frac{1}{\abs{Q}}\sum_{q∈Q}{\frac{\abs{\tilde{N}_{k}(q)∩N_{k}(q)}}{k}}`, "recall"),
  P(R`where $Q$ is the set of queries. Recall measures how close the ANN result is to linear search, not whether a user finds the images relevant. To measure the modality gap we use $cos(μ_{I},μ_{T})$, the cosine between the mean image vector $μ_{I}$ and the mean text vector $μ_{T}$. Table 1 summarises the methods.`),
  ...table(1, "The compared methods. $n$: database size, $d$: dimension (512), $b$: LSH bits (number of hashing functions), $M$: number of PQ subspaces or HNSW links per node, $K$: centroids per subspace (256), $ef$: HNSW search list size.",
    ["Method", "Family", "Learns from data", "Bytes per vector", "Query cost", "Parameter swept"],
    [["LSH (Charikar, 2002)", "Hashing", "No", "b / 8", "O(nb) bit ops", "b = 64 … 4,096"],
     ["PQ (Jégou et al., 2011)", "Quantization", "Yes (k-means)", "M", "O(Kd + nM)", "M = 8, 16, 32, 64"],
     ["HNSW (Malkov & Yashunin, 2020)", "Graph", "Yes (graph)", "4d + 8M", "≈ O(ef·M·d·log n)", "ef = 100 … 1,024"],
     ["Linear search (exact)", "-", "-", "4d", "O(nd)", "-"]],
    [2.6, 1.3, 1.4, 1.3, 1.9, 1.9]),

  H2("2.2", "LSH with random projection"),
  P(R`As in the lecture, we generate $b$ hashing functions by drawing $b$ random vectors $w_{1},…,w_{b}$ from a standard Gaussian distribution. Each hashing function turns an input vector into one bit ($1[·]$ is 1 if the condition holds and 0 otherwise):`),
  EQ(R`h_{i}(x)=1\b{w_{i}^{⊤}x>0}, H(x)=(h_{1}(x),…,h_{b}(x))∈\c{0,1}^{b}`, "hash"),
  P(R`The $b$ bits form the binary vector of $x$. The whole database is converted into binary vectors once (offline), and at query time only the query is converted (online). We then run linear search on the binary vectors using the Hamming distance, which only needs an XOR and a bit count, so it is much cheaper than a distance on float values. A 512-bit binary vector takes 64 bytes, 32 times less than the 2,048-byte float vector. We use LSH + linear search, implemented with IndexLSH from the FAISS library (Douze et al., 2024), instead of hash tables because it returns a full top-100 list. Results are averaged over three random seeds (standard deviation of recall below 0.01).`),
  P(R`**Why more bits give better results.** A random hyperplane separates two vectors with a probability equal to the angle between them divided by $π$ (Charikar, 2002):`),
  EQ(R`Pr\b{h_{i}(q)≠h_{i}(x)}=\frac{θ(q,x)}{π} ⇒ E\b{D_{H}(q,x)}=\frac{b θ(q,x)}{π}`, "charikar"),
  P(R`So the Hamming distance $D_{H}$ estimates the angle, and the estimate gets more accurate as $b$ grows. This is why, as shown in the lecture, more hashing functions give more accurate top-$k$ results. We can make this precise. Let $u$ be a true neighbour and $v$ a non-neighbour, with angular margin $δ=(θ(q,v)−θ(q,u))/π>0$. Treating the bits as independent, the probability that the binary vectors rank $v$ above $u$ is approximately`),
  EQ(R`Pr[swap]≈Φ(−z), z=δ\sqrt{\frac{b}{θ(u,v)/π−δ^{2}}}`, "swap"),
  P(R`where $Φ$ is the standard normal CDF. Since $z$ grows with $δ\sqrt{b}$, halving the margin needs four times as many bits for the same accuracy.`),

  H2("2.3", "Product quantization"),
  P(R`PQ (Jégou et al., 2011) follows the steps in the lecture. (1) Split each vector equally into $M$ sub-vectors, one per subspace. (2) In each subspace, run k-means on the database sub-vectors to get $K=256$ centroids $c_{m,1},…,c_{m,K}$. (3) Replace each sub-vector with the ID of its nearest centroid. Each vector is stored as $M$ cluster IDs of one byte each (the compressed vector), so with $M=64$ a 2,048-byte vector becomes 64 bytes, and it is approximated by $\hat{x}$, its centroids put back together.`),
  P(R`At query time we use the asymmetric distance: only the database is compressed and the query stays exact. For each subspace the query fills a look-up table $T_{m}[j]=\a{q^{(m)},c_{m,j}}$ of its similarity to the 256 centroids. The approximate score of a database item is then the sum of $M$ table entries:`),
  EQ(R`\hat{s}(q,x)=\a{q,\hat{x}}=\sum_{m=1}^{M}{T_{m}\b{i_{m}(x)}}, s(q,x)−\hat{s}(q,x)=\a{q,x−\hat{x}}`, "adc"),
  P(R`where $i_{m}(x)$ is the cluster ID of $x$ in subspace $m$. (The lecture uses the squared L2 distance; for unit vectors it gives the same ranking.) The second part of Eq. (${EN.adc}) matters later. k-means minimises the *reconstruction* error $\n{x−\hat{x}}^{2}$, but the ranking depends on the *score* error $\a{q,x−\hat{x}}$, which depends on the queries. If the queries have second-moment matrix $S=E[qq^{⊤}]$, the average squared score error is`),
  EQ(R`E_{q}\b{\a{q,x−\hat{x}}^{2}}=(x−\hat{x})^{⊤}S(x−\hat{x})`, "scoreerr"),
  P(R`This equals the k-means objective only when $S∝I$. Image and text queries have very different $S$, so centroids fitted to images are not the best choice for captions (Section 2.7). We also test **OPQ** (Ge et al., 2014), which rotates the vectors before PQ so that the variance is spread evenly across the subspaces. A rotation does not change inner products.`),

  H2("2.4", "Hierarchical navigable small world graphs (HNSW)"),
  P(R`HNSW (Malkov & Yashunin, 2020) is a graph index from outside the course. Each vector is a node linked to up to $M$ similar nodes ($2M$ on the bottom layer). The graph has several layers, and each layer keeps a random $1/M$ of the nodes in the layer below, so the upper layers act as long-range shortcuts. A search starts at the top layer and keeps moving to the neighbour most similar to the query until no neighbour is better, then goes down one layer. On the bottom layer it keeps a list of the $ef$ best candidates and stops when no remaining candidate can improve the list. The search cost grows roughly with $log n$, and $ef$ trades speed for recall without rebuilding the graph. We use $M=32$ and $efConstruction=200$ (the list size used while building). HNSW stores the full vectors plus the links, about 2,320 bytes per vector, which is more than the raw data.`),

  H2("2.5", "Linear search"),
  P("Exact linear search (FAISS IndexFlatIP) gives the ground truth, the true top-100 of every query, and is our speed baseline."),

  H2("2.6", "Why text queries are harder"),
  P(R`LSH and PQ rank the database by an approximate score $\hat{s}(q,x)=s(q,x)+ε(q,x)$. A true neighbour $u$ is pushed out of the top-$k$ by a non-neighbour $v$ when`),
  EQ(R`ε(q,v)−ε(q,u)>s(q,u)−s(q,v)`, "displace"),
  P(R`The right-hand side depends only on the query. It is small when the top-$k$ neighbours have almost the same scores as the items just below them. We measure this by the **neighbour contrast** $γ(q)=s_{(1)}(q)−s_{(100)}(q)$, where $s_{(r)}(q)$ is the similarity to the $r$-th true neighbour. If the score error has size $σ_{ε}(q)$, what matters is the ratio`),
  EQ(R`ρ(q)=\frac{σ_{ε}(q)}{γ(q)}`, "rho"),
  P(R`so halving the contrast is as bad as doubling the error. HNSW is different: it computes exact scores, but only for the nodes it visits. If the query is far from all images, all similarities are small and close to each other, so the greedy search gets little guidance and may stop in the wrong region. For HNSW, the property that matters is the similarity to the nearest image, $s_{(1)}(q)$.`),
  P("This gives two predictions. **(P1)** LSH and PQ lose recall on queries with low contrast $γ$, whether or not they learn from the data. **(P2)** HNSW loses recall on queries with low $s_{(1)}$, and more so on larger databases, where the search path is longer."),

  H2("2.7", "Improving the methods for text queries"),
  P("Each improvement uses a sample of training captions that does not overlap with the test queries, and none of them changes the exact ground truth."),
  P(R`**Modality-aware centring for LSH.** CLIP image vectors lie in a narrow cone around their mean $μ_{I}$. Many random hyperplanes do not cut through this cone, so every image gets the same bit and the bit is wasted. A common fix is to subtract the image mean before hashing, $h_{i}(x)=1[w_{i}^{⊤}(x−μ_{I})>0]$ (using the median of each projection as the threshold gives 99% the same bits on COCO). But for a text query the offset $μ_{T}−μ_{I}$ then dominates, so the query lands on the same side of most hyperplanes whatever its content. We therefore centre each modality on its own mean:`),
  EQ(R`h_{i}(x)=1\b{w_{i}^{⊤}(x−μ_{I})>0} (database), h_{i}(q)=1\b{w_{i}^{⊤}(q−μ_{T})>0} (text queries)`, "lshqa"),
  P(R`with $μ_{T}$ estimated from the training captions. The database and image queries are hashed as before, so image-query results do not change and there is no extra build cost.`),
  P(R`**Query-weighted PQ.** Eq. (${EN.scoreerr}) says that the error that matters is weighted by $S$. Let $A=S^{1/2}$. If we index $x'=Ax$ and search with $q'=A^{−1}q$, every inner product stays the same, $\a{q',x'}=\a{q,x}$, so the ground truth does not change. But now ordinary k-means minimises the average score error for these queries:`),
  EQ(R`\n{x'−\hat{x}'}^{2}=(x−\hat{x})^{⊤}S(x−\hat{x})=E_{q}\b{\a{q,x−\hat{x}}^{2}}`, "qa2"),
  P(R`We estimate $S$ from 200,000 training captions (*text-weighted*) or as the average of the caption and image statistics (*mixed*), add a small constant to its eigenvalues for stability, and combine it with the OPQ rotation. The extra query cost is one $512×512$ matrix–vector product.`),
  P(R`**Caption-guided edges for HNSW.** HNSW links images that are similar to each other. The true neighbours of a caption, however, may not be similar to each other: a street scene and a close-up of a bus can both match “a red bus in a city”. Following RoarGraph and OOD-DiskANN, we add edges between images that are relevant to the same caption (Algorithm 1).`),
  ...algorithm("**Algorithm 1.** Caption-guided edge insertion", [
    [0, "**Input:** HNSW graph $G$ with bottom-layer capacity $2M$ per node; image vectors $X$; training captions $T$; $K=16$; per-node cap $E_{max}=16$"],
    [0, "**for each** caption $t∈T$ **do**"],
    [1, "$(u_{1},…,u_{K})←$ exact top-$K$ images of $t$ in $X$"],
    [1, "propose edges $(u_{1},u_{j})$ and $(u_{j},u_{1})$ for $j=2,…,K$"],
    [0, "count how many captions propose each edge; drop edges already in $G$"],
    [0, "**for each** node $u$ **do**"],
    [1, "add $u$'s proposed edges to its bottom-layer list, most-proposed first, while free slots remain and at most $E_{max}$ are added"],
    [0, "**Output:** $G'$, with the same memory and the same search procedure"],
  ]),
  P(R`The new edges go into bottom-layer slots that HNSW has allocated but left empty (the average degree is 21.6 on COCO and 23.1 on LAION, out of 64), so memory and the search code do not change. As ablations we also try two other uses of the captions: *caption entry points*, which start the search from the true neighbours of the most similar training captions, and *captions as graph nodes*, which insert the captions into the graph as extra nodes that are never returned.`),
);

// ---- 3 experiments
add(H1("3", "Experiments"),
  H2("3.1", "Experimental setup"),
  ...table(2, "Datasets. Both use CLIP ViT-B/32 embeddings (512-d, unit norm).",
    ["", "COCO", "LAION"],
    [["Source", "MS-COCO 2014 train + val, embedded by us", "VIBE benchmark (LAION-400M subset)"],
     ["Database (indexed)", "122,287 images", "999,448 images"],
     ["Image queries", "1,000 held-out images", "1,000 held-out images"],
     ["Text queries", "First caption of the same 1,000 images", "1,000 VIBE web-text queries (unpaired)"],
     ["Training captions", "122,287 captions of database images", "999,448 VIBE training queries"],
     ["Ground truth", "Exact top-100 by linear search", "Exact top-100 (agrees with VIBE's to ≥ 0.999)"]],
    [2, 3.5, 3.5]),
  P("**Datasets.** COCO (Lin et al., 2014) and LAION (Schuhmann et al., 2021) differ in size (8×) and in the kind of text: clean human-written captions on COCO and noisy web text on LAION (Table 2). An image query is a held-out image that is not in the database. It searches the same image database (image-to-image search) and serves as the in-distribution baseline. On COCO the image and text queries describe the same 1,000 images, so the only difference between the two query sets is which CLIP encoder produced them."),
  P("**Protocol.** We follow VIBE (Jääsaari et al., 2026) and ann-benchmarks (Aumüller et al., 2020). (i) Indexes are built from the database only and never see a test query. (ii) Each index is built once and searched with both image and text queries using the same parameters, so any difference comes from the queries. (iii) Queries are sent one at a time on a single CPU thread, and we report the fastest of several runs. All runs used one server (2× AMD EPYC 7742) with FAISS 1.15.1 and fixed random seeds."),
  P(R`**Metrics.** Our main metric is recall@100 (Eq. ${EN.recall}). We also report recall@10, queries per second (QPS), index size in bytes per vector and build time. The gap between the two query types, on the same index with the same setting, is`),
  EQ(R`Δ=Recall@100(Q_{image})−Recall@100(Q_{text})`, "delta"),
  P("We never average image and text recall, because their difference is what we study."),

  H2("3.2", "How text and image queries differ"),
  ...table(3, "How the queries relate to the indexed images. $γ$ is the neighbour contrast from Section 2.6.",
    ["Dataset", "Query type", "$cos$(query mean, image mean)", "$s_{(1)}$", "$s_{(100)}$", "Contrast $γ$"],
    [["COCO", "image", "0.999", "0.858", "0.771", "0.087"], ["COCO", "text", "0.309", "0.345", "0.299", "0.046"],
     ["LAION", "image", "0.999", "0.825", "0.733", "0.092"], ["LAION", "text", "0.357", "0.355", "0.300", "0.055"]],
    [1.2, 1.2, 2.4, 1.3, 1.3, 1.5]),
  ...figure("fig1_neighbour_similarity.png", 1720, 681, 1, "Similarity of each query to its 1st to 100th true neighbour (solid: image queries; dashed: text queries; band: 10th to 90th percentile)."),
  P("Table 3 confirms the modality gap: the mean text query points in a very different direction from the mean image (cosine 0.31 to 0.36), while the mean image query matches it almost exactly (0.999). Figure 1 shows the two properties from Section 2.6. Text queries are far from the data: their most similar image scores 0.35, against 0.86 for an image query. Their neighbours are also bunched together: the contrast $γ$ is about half that of image queries (0.046 vs 0.087 on COCO), so by Eq. (" + EN.rho + ") the same scoring error does about twice as much damage."),

  H2("3.3", "Comparison of the three methods"),
  ...figure("fig2_recall_vs_qps.png", 1722, 854, 2, "Recall@100 vs single-thread queries per second. Colour shows the method; solid lines and filled markers are image queries, dashed lines and hollow markers are text queries; the star is linear search."),
  ...figure("fig3_recall_vs_memory.png", 1685, 854, 3, "Recall@100 vs index size in bytes per vector."),
  ...table(4, "Best setting of each method with image queries.",
    ["Dataset", "Method", "Setting", "Recall@100", "QPS", "Speed-up", "Bytes / vector", "Build (s)"],
    [["COCO", "HNSW", "$ef$ = 128", "0.992", "2,235", "30×", "2,320", "26"],
     ["COCO", "PQ", "$M$ = 64", "0.657", "181", "2.5×", "68", "5"],
     ["COCO", "LSH", "$b$ = 4,096", "0.837", "128", "1.7×", "581", "13"],
     ["COCO", "Linear", "-", "1", "74", "1×", "2,048", "0.2"],
     ["LAION", "HNSW", "$ef$ = 512", "0.984", "488", "51×", "2,320", "226"],
     ["LAION", "PQ", "$M$ = 64", "0.560", "22", "2.3×", "65", "12"],
     ["LAION", "LSH", "$b$ = 4,096", "0.805", "17", "1.8×", "520", "72"],
     ["LAION", "Linear", "-", "1", "9.6", "1×", "2,048", "1.3"]],
    [1.1, 1, 1.4, 1.2, 1, 1.1, 1.3, 1]),
  P("With image queries, each method behaves as its design suggests (Figures 2 and 3, Table 4). **HNSW** is the fastest and most accurate, with recall@100 of at least 0.98 at 30 to 51 times the speed of linear search, because it computes only 1,400 to 2,000 similarities per query instead of $n$. The price is memory (13% *more* than the raw vectors) and build time (226 s for 1M vectors on 32 threads). **PQ** is the opposite: it is 32 to 240 times smaller than the raw vectors, but on its own it reaches recall@100 of at most 0.66, and it is not much faster because it still scans every compressed vector. **LSH** needs no training and builds in seconds, but needs 4,096 bits (512 bytes) to reach recall 0.80 to 0.84, and at that length its scan is barely faster than linear search."),
  ...table(5, "Recall@100 (image / text) at equal bits per vector; PQ uses $M=b/8$.",
    ["Bits / vector", "COCO LSH", "COCO PQ", "LAION LSH", "LAION PQ"],
    [["64", "0.155 / 0.011", "0.303 / 0.179", "0.075 / 0.002", "0.111 / 0.059"],
     ["128", "0.270 / 0.025", "0.392 / 0.229", "0.187 / 0.008", "0.198 / 0.114"],
     ["256", "0.420 / 0.071", "0.505 / 0.336", "0.337 / 0.030", "0.345 / 0.232"],
     ["512", "0.577 / 0.181", "0.657 / 0.515", "0.504 / 0.113", "0.560 / 0.438"]],
    [1.2, 2, 2, 2, 2]),
  P("Table 5 compares LSH and PQ at the same code length. PQ's learned centroids beat LSH's random hyperplanes at every size: slightly for image queries, but by 2.8 to 30 times for text queries. LSH, however, scans 5 to 10 times faster, because a Hamming distance is cheaper than $M$ table look-ups. This is the data-independent versus data-dependent trade-off from the lecture."),

  H2("3.4", "The cost of querying with text"),
  ...figure("fig4_param_sweep_gap.png", 2498, 1222, 4, "Image (solid) and text (dashed) recall@100 across each method's parameter. The shaded area is the gap Δ."),
  ...table(6, "The gap Δ = recall@100(image) − recall@100(text) at both ends of each parameter sweep, and the largest Δ in the sweep.",
    ["Dataset", "Method", "Smallest setting", "Largest setting", "Largest Δ"],
    [["COCO", "HNSW", "$ef$ 100: 0.986 vs 0.884 (Δ 0.101)", "$ef$ 1024: 1.000 vs 0.996 (Δ 0.004)", "0.101"],
     ["COCO", "PQ", "$M$ 8: 0.303 vs 0.179 (Δ 0.123)", "$M$ 64: 0.657 vs 0.515 (Δ 0.142)", "0.169"],
     ["COCO", "LSH", "$b$ 64: 0.155 vs 0.011 (Δ 0.144)", "$b$ 4096: 0.837 vs 0.559 (Δ 0.279)", "0.395"],
     ["LAION", "HNSW", "$ef$ 100: 0.939 vs 0.749 (Δ 0.190)", "$ef$ 1024: 0.988 vs 0.956 (Δ 0.031)", "0.190"],
     ["LAION", "PQ", "$M$ 8: 0.111 vs 0.059 (Δ 0.053)", "$M$ 64: 0.560 vs 0.438 (Δ 0.122)", "0.122"],
     ["LAION", "LSH", "$b$ 64: 0.075 vs 0.002 (Δ 0.072)", "$b$ 4096: 0.805 vs 0.515 (Δ 0.290)", "0.401"]],
    [1.1, 1, 2.7, 2.7, 1.1]),
  P("Every method loses recall on text queries at every setting (Figure 4, Table 6). The parameters behave differently. For HNSW, a larger $ef$ closes most of the gap (COCO: 0.10 to 0.004; LAION: 0.19 to 0.03), but at a cost: to match the image-query recall at $ef$ = 100 to 128, text queries need $ef$ ≈ 490 to 900, which makes search 3.4 to 5.3 times slower. For PQ, more subspaces do not help, and Δ stays between 0.12 and 0.17 on COCO. LSH has the largest gap, up to 0.40. This is the opposite of what we expected: LSH learns nothing from the images, so its loss must come from the text queries themselves."),

  H2("3.5", "Checking the explanation"),
  ...figure("fig5_which_queries_fail.png", 1982, 1262, 5, "Per-query recall@100, with queries grouped into 12 equal-size bins. Left: HNSW against the query's similarity to its nearest image, $s_{(1)}$. Right: PQ against the neighbour contrast $γ$."),
  P(R`**(P1) Low contrast hurts PQ and LSH.** For PQ, per-query recall increases with contrast (Spearman correlation up to +0.51; Figure 5, right), and 39% of COCO text queries get recall below 0.5, against 3% of image queries. The PQ score error is actually *smaller* for text queries (RMS 0.038 vs 0.052 on COCO at $M=64$). The difference is the contrast, so the ratio $ρ$ of Eq. (${EN.rho}) is 0.78 for text and 0.57 for images.`),
  P(R`For LSH, we first checked that the measured probability of two bits differing matches Charikar's $θ/π$ within 0.001 for *both* query types, so the hashing functions treat text and images the same way. The difference is again in the margins: on LAION, the median angle between a query's 100th and 1,000th neighbours is 4.7° for image queries but only 1.8° for text queries. Table 7 shows that Eq. (${EN.swap}) predicts the measured swap rate well.`),
  ...table(7, "LSH on LAION: probability that the 1,000th neighbour is ranked above the 100th, predicted by Eq. (" + EN.swap + ") and measured (seed 0).",
    ["Query type", "Margin", "$b$", "$z$", "Predicted", "Measured", "Recall@100"],
    [["image", "4.7°", "256", "0.82", "0.199", "0.169", "0.337"],
     ["image", "4.7°", "512", "1.16", "0.126", "0.104", "0.504"],
     ["image", "4.7°", "4,096", "3.28", "0.006", "0.007", "0.805"],
     ["text", "1.8°", "256", "0.30", "0.380", "0.349", "0.030"],
     ["text", "1.8°", "512", "0.42", "0.333", "0.333", "0.113"],
     ["text", "1.8°", "4,096", "1.19", "0.123", "0.091", "0.517"]],
    [1.2, 1, 1, 1, 1.2, 1.2, 1.3], "The model slightly overestimates the swap rate because FAISS uses orthogonal, not independent, hyperplanes."),
  P(R`Since $z∝δ\sqrt{b}$, a margin 2.6 times smaller needs $2.6^{2}≈7$ times more bits. Indeed, text queries at 4,096 bits ($z=1.19$, recall 0.517) behave like image queries at 512 bits ($z=1.16$, recall 0.504), which matches the shift between the two LSH curves in Figure 4. LSH suffers most because it cannot adapt: unlike PQ, nothing in its construction puts more precision where the text queries need it.`),
  P(R`**(P2) Distance from the data hurts HNSW.** HNSW's per-query recall increases with the query's similarity to its nearest image (Spearman correlation +0.55 on COCO and +0.64 on LAION; Figure 5, left), and text queries sit at the low end. On LAION at $ef=100$, text queries even compute 1% *fewer* similarities than image queries, yet lose 19 points of recall: the search stops early in the wrong region, as predicted.`),

  H2("3.6", "Ablation studies"),
  H3("HNSW build parameters"),
  ...figure("fig7_hnsw_build_params.png", 1701, 674, 6, "HNSW recall@100 against graph degree $M$ (left) and $efConstruction$ (right), COCO."),
  P("More links per node reduce the gap (Figure 6). Going from $M=8$ to $M=64$ raises COCO text recall at $ef=100$ from 0.79 to 0.89 and lowers Δ from 0.155 to 0.098, at 21% more memory, and most of the gain is reached at $M=16$ to 32. $efConstruction$ has no consistent effect (text recall 0.92, 0.86, 0.88 and 0.90 for 40, 100, 200 and 400): a more careful build connects similar images better, which is not what text queries need."),
  H3("Database size"),
  ...figure("fig8_scaling.png", 1701, 717, 7, "Recall@100 against database size (random LAION subsets). Left: HNSW; right: PQ."),
  P("Using random LAION subsets with recomputed ground truth (Figure 7), HNSW's gap at $ef=100$ grows from 0.10 at 10k images to 0.13 at 100k and 0.19 at 1M, as (P2) predicts: a longer search path has more chances to stop in the wrong place. PQ's gap stays at 0.10 to 0.12 at every size."),
  H3("Removing the offset between modalities"),
  ...table(8, "Effect of centring each modality on its own mean (recall@100 image / text, and Δ).",
    ["Dataset", "Method", "Original space", "Centred space"],
    [["COCO", "HNSW, $ef$ = 100", "0.986 / 0.884 (Δ 0.101)", "0.987 / 0.931 (Δ 0.057)"],
     ["COCO", "PQ, $M$ = 64", "0.656 / 0.512 (Δ 0.143)", "0.738 / 0.573 (Δ 0.165)"],
     ["LAION", "HNSW, $ef$ = 100", "0.939 / 0.749 (Δ 0.190)", "0.964 / 0.868 (Δ 0.096)"],
     ["LAION", "PQ, $M$ = 64", "0.561 / 0.437 (Δ 0.124)", "0.678 / 0.514 (Δ 0.164)"]],
    [1.1, 2, 3, 3]),
  P(R`We subtract $μ_{I}$ from the images and $μ_{T}$ from the text queries, normalise again, and rebuild every index *and the ground truth* in this space (Table 8). This halves HNSW's gap, supporting (P2). For PQ it doubles the contrast of both query types, so both recalls rise, but text queries still have half the contrast of images and the gap stays, supporting (P1). Centring cannot be used in practice, because it changes the answer: only 62 to 72% of a text query's original top-100 is still in its new top-100.`),

  H2("3.7", "Results of the improved methods"),
  P("All results here use the original exact top-100, so they compare directly with the sections above."),
  H3("HNSW with caption-guided edges"),
  ...table(9, "HNSW variants at $ef$ = 100 (recall@100). Variants 1 and 2 are the ablations from Section 2.7. Bold: best per dataset; underlined: second best (COCO only).",
    ["Dataset", "Variant", "Image", "Text", "Δ", "Extra memory"],
    [["COCO", "Baseline", "__0.986__", "__0.884__", "0.101", "-"],
     ["COCO", "1. Caption entry points", "__0.986__", "0.877", "0.108", "+2,328 B/vector"],
     ["COCO", "2. Captions as graph nodes", "0.985", "0.078", "0.907", "+232 B/vector"],
     ["COCO", "3. Caption-guided edges (Alg. 1)", "**0.987**", "**0.947**", "0.040", "none"],
     ["LAION", "Baseline", "0.939", "0.749", "0.190", "-"],
     ["LAION", "3. Caption-guided edges (Alg. 1)", "**0.950**", "**0.818**", "0.132", "none"]],
    [1.1, 3.2, 1, 1, 1, 1.7]),
  ...figure("fig9_hnsw_fixes.png", 1741, 879, 8, "HNSW variants: recall@100 against QPS (dashed: text queries). Purple: caption-guided edges (lighter shades use fewer training captions); green: caption entry points."),
  P("Better entry points (variant 1) do not help, so the problem is not where the search starts. Captions as nodes (variant 2) make things much worse: a caption node is far more similar to a text query (about 0.8) than any image is (about 0.3), so caption nodes fill the whole candidate list and the search never reaches the images. Caption-guided edges (variant 3) work (Table 9, Figure 8). On COCO, text recall rises from 0.88 to 0.95, Δ drops by 60%, and image recall also rises slightly. At equal recall the modified graph is 1.5 times faster (text recall 0.947 at 2,170 QPS against 0.948 at 1,470 QPS). On LAION, text recall improves by 7 points at $ef=100$ when all training captions are used, but not with only 10% of them. The offline cost is an exact top-16 search per training caption: 73 s for COCO and 1.7 h on 32 cores for LAION."),

  H3("Query-weighted PQ"),
  ...table(10, "PQ variants at $M$ = 64 bytes per vector: recall@100 (image / text, Δ) and error-to-contrast ratio $ρ$ (image / text). Bold: best per column and query type; underlined: second best.",
    ["Variant", "Weights", "OPQ", "COCO recall", "LAION recall", "COCO $ρ$", "LAION $ρ$"],
    [["PQ (baseline)", "none", "no", "0.656 / 0.512 (Δ 0.143)", "0.561 / 0.437 (Δ 0.124)", "0.57 / 0.78", "0.72 / 0.82"],
     ["OPQ", "none", "yes", "__0.773__ / 0.665 (Δ 0.108)", "__0.662__ / __0.537__ (Δ 0.124)", "**0.26** / 0.36", "**0.47** / __0.56__"],
     ["Text-weighted PQ", "captions", "no", "0.160 / 0.461", "0.124 / 0.392", "0.99 / 0.82", "0.95 / 0.90"],
     ["Text-weighted OPQ", "captions", "yes", "0.632 / **0.793** (Δ −0.161)", "0.554 / **0.579** (Δ −0.025)", "__0.43__ / **0.21**", "0.65 / **0.53**"],
     ["Mixed OPQ", "50/50", "yes", "**0.809** / __0.732__ (Δ 0.077)", "**0.691** / 0.519 (Δ 0.171)", "**0.26** / __0.26__", "__0.50__ / 0.60"]],
    [1.6, 1.0, 0.75, 2.35, 2.35, 1.1, 1.1]),
  ...figure("fig10_pq_fixes.png", 1983, 1302, 9, "PQ variants: recall@100 against code size, for image queries (top) and text queries (bottom)."),
  P("The OPQ rotation is necessary (Table 10, Figure 9). Weighting alone puts most of the variance into a few directions, which a fixed split into sub-vectors handles badly, and image recall falls to 0.12 to 0.16. With the rotation, text weighting raises COCO text recall from 0.51 to 0.79 at the same 64 bytes, and on LAION it removes the gap at every code size. If both query types matter, mixed weighting is the better choice: on COCO it gives the best image recall, the second-best text recall, and halves Δ. The ratio $ρ$ ranks the variants in almost the same order as recall, which supports Section 2.6: what raises recall is a lower score error relative to the contrast, not a lower reconstruction error."),
  P("**Re-ranking.** Re-scoring PQ's top 1,000 candidates with the full vectors (at most 21% slower) raises recall at $M$ = 64 to 0.998 / 0.974 on COCO and 0.968 / 0.895 on LAION. But it needs the full vectors in memory or on disk, which removes PQ's memory advantage, and it cannot recover neighbours missing from the candidate list. Better codes and re-ranking therefore work well together."),

  H3("Modality-aware LSH"),
  ...figure("fig11_lsh_variants.png", 1983, 682, 10, "LSH variants: text-query recall@100 against code length. Median thresholds and image-mean centring overlap."),
  P(R`Figure 10 shows text queries only, since image queries are hashed the same way in all centred variants. The two image-based fixes, median thresholds and image-mean centring, make text recall much worse: at 4,096 bits it falls from 0.56 to 0.22 on COCO and from 0.52 to 0.07 on LAION, because a text query lands on the same side of most hyperplanes, as predicted in Section 2.7. Modality-aware centring (Eq. ${EN.lshqa}) fixes this and beats plain LSH. Text recall roughly doubles at medium code lengths (COCO, 512 bits: 0.18 to 0.33; LAION, 1,024 bits: 0.23 to 0.40), and the LAION gap at 4,096 bits falls from 0.29 to 0.12.`),

  H2("3.8", "Success and failure cases"),
  ...figure("fig12_examples.png", 1722, 1257, 11, "Three COCO test captions. Rows: exact top-10, HNSW ($ef$ = 100), HNSW with caption-guided edges, PQ ($M$ = 64). Green border: in the exact top-10; red: not."),
  P("Figure 11 shows three COCO captions chosen to show the two kinds of failure. For the long caption about a train, HNSW gets stuck in a nearby region and drifts to lorries and car parks (recall@10 = 0.5), while the graph with caption-guided edges returns exactly the true top-10. For the baseball caption, PQ finds the right topic but the wrong order: every result is a reasonable baseball photo, but none is in the exact top-10, because hundreds of images have almost the same score and small errors reshuffle them. A user might still be happy with these results, which is a limitation of measuring recall against linear search. Over all captions, HNSW usually succeeds (text recall@10 of 0.949, and 0.987 with caption-guided edges)."),

  H2("3.9", "Discussion"),
  ...table(11, "Strengths and weaknesses of the three methods, based on our measurements.",
    ["Method", "Strengths", "Weaknesses", "Best use"],
    [["HNSW", "Recall@100 of 0.98 to 0.99 at 30 to 51× linear-search speed; tunable at query time; text loss shrinks with $ef$ and caption-guided edges", "More memory than the raw vectors; slow to build; text loss grows with database size", "Low-latency search when there is enough RAM"],
     ["PQ", "32 to 240× smaller; fast to build; query-weighted OPQ recovers text recall", "Recall at most 0.66 on its own; scan barely faster than linear search; gap does not shrink with more subspaces", "Memory-limited systems, as a first stage before re-ranking"],
     ["LSH", "No training; builds in seconds; scans 5 to 10× faster than PQ at equal bits; simple error model", "Needs 4,096 bits for recall 0.8; largest text gap (up to 0.40); image-based centring destroys text recall", "Streaming data and quick baselines, with modality-aware centring"]],
    [1, 3.2, 3.2, 2.2]),
  P("**Key factors.** Two separate factors cause the text-query loss. The first is low neighbour contrast, a property of the queries, which is why it affects even LSH. Longer codes reduce it only slowly ($z∝δ\\sqrt{b}$); what helps is putting the code's precision where the queries need it, with query-weighted centroids or hash functions centred on each modality. The second is distance from the data, which only affects graph search, grows with database size, and is reduced by adding edges between images relevant to the same caption."),
  P("**Parameter settings.** For LSH, the number of bits $b$ is the only setting, and its benefit grows only as $\\sqrt{b}$. For PQ, $M$ trades memory for accuracy, but for text queries the rotation and weighting matter more than doubling $M$. For HNSW, $ef$ is the most useful setting at query time, $M$ helps up to about 32, and $efConstruction$ has no reliable effect on text recall."),
  P("**Limitations.** We use only one encoder (CLIP ViT-B/32); newer encoders may have a smaller modality gap. Timing is single-threaded on CPU, and independent jobs shared one server, which may lower absolute QPS but does not affect recall. Recall is measured against linear search, not human judgement (Figure 11). The improved methods need training captions from the same distribution as the real queries, and we did not test what happens when they differ."),
);

// ---- 4 conclusion
add(H1("4", "Conclusion"),
  P("We compared LSH, PQ and HNSW for text-to-image search with CLIP on COCO (122k images) and LAION (1M images), searching each index with both image and text queries. With image queries we see the usual trade-off: HNSW is the fastest and most accurate, PQ is the smallest, and LSH is the cheapest to build. With text queries every method loses recall, and LSH loses the most even though it does not learn from the data."),
  P("Two properties of text queries explain this. Their top-$k$ neighbours have very similar scores, so the scoring errors of LSH and PQ reorder them; for LSH a simple probability model matches the measured errors and shows why text queries need about eight times more bits. Text queries are also far from every image, so HNSW's greedy search stops in the wrong place, more often on larger databases. Using this, we improved each method while keeping the exact ground truth: caption-guided edges for HNSW (COCO text recall 0.88 to 0.95, no extra memory), query-weighted OPQ (PQ text recall 0.51 to 0.79 at 64 bytes), and modality-aware centring for LSH (text recall roughly doubled at medium code lengths). Our main lesson is that indexes for cross-modal search should be built, tuned and evaluated with the kind of queries they will actually receive."),
);

// ---- references
add(H1("", "References"),
  ...[
    "M. Aumüller, E. Bernhardsson, A. Faithfull. ANN-Benchmarks: a benchmarking tool for approximate nearest neighbor algorithms. Information Systems 87, 2020.",
    "M. S. Charikar. Similarity estimation techniques from rounding algorithms. STOC, 2002.",
    "M. Chen et al. RoarGraph: a projected bipartite graph for efficient cross-modal approximate nearest neighbor search. VLDB, 2024.",
    "M. Douze et al. The Faiss library. arXiv:2401.08281, 2024.",
    "T. Ge, K. He, Q. Ke, J. Sun. Optimized product quantization. IEEE TPAMI 36(4), 2014.",
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
  creator: "SC4020 Group 21",
  title: "SC4020 Project 1: Technical Review Report - Similarity Search",
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
    headers: { default: new Header({ children: [small([new TextRun({ text: "SC4020 Project 1 Report: Similarity Search", size: 17, italics: true, color: "555555" })])] }) },
    footers: { default: new Footer({ children: [small([new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "555555" })])] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
