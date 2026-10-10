// All project content, taken from TEAM_95_PR2.pdf and Review_2.pdf.
// Items that are not in either PDF are marked TODO_* — replace them before publishing.

export const LINKS = {
  github: 'https://github.com/SuryaSampreeth/VLIL-Team-95',
  video: 'TODO_DEMO_VIDEO_URL', // YouTube embed URL (https://www.youtube.com/embed/<id>) or /video/demo.mp4
}

export const isTodo = (v) => !v || String(v).startsWith('TODO_')

export const PROJECT = {
  short: 'ReXTrust',
  title: 'ReXTrust: XAI-Driven Multi-Disease Diagnosis in Chest X-rays via Conditional Enhancement and Grounded Reporting',
  tagline: 'One pipeline from a raw chest X-ray to a diagnosis you can actually trust.',
  type: 'Type 1: Research Project',
  team: 'Team 95',
  department: 'Department of CSE',
  guide: { name: 'Dr. Anupa Vijai', role: 'Assistant Professor, Department of CSE' },
}

export const PROBLEMS = [
  {
    title: 'Degraded Image Quality',
    body: 'X-rays from low-resource settings suffer from noise, low contrast, and poor illumination, reducing diagnostic reliability.',
    icon: 'ScanEye',
  },
  {
    title: 'Lack of Per-Disease Spatial Verification',
    body: 'Existing explainability methods produce blended saliency maps, making it difficult to localize individual diseases.',
    icon: 'Layers',
  },
  {
    title: 'Ungrounded AI-Generated Reports',
    body: 'Vision-language models generate fluent clinical text without verification, leading to potential hallucinated diagnoses.',
    icon: 'FileWarning',
  },
]

export const OBJECTIVES = [
  {
    title: 'Enhance only when needed',
    body: 'Automatically judge whether an X-ray is already clear enough, and only improve the ones that genuinely need it — without losing important medical details.',
  },
  {
    title: 'Detect diseases with a proven model',
    body: 'CNN, Transformer, and graph-based models were benchmarked under the exact same conditions. The pipeline then runs one fixed detection model, chosen on that evidence — not because it got an easier setup.',
  },
  {
    title: 'Show proof for every disease, separately',
    body: 'For each disease the model predicts, show exactly where in the image that prediction came from — one clear picture per disease, not one mixed map.',
  },
  {
    title: 'Write a report, then check it',
    body: 'Generate a radiology report in plain language, then verify every claim in it before treating it as final.',
  },
]

export const GAPS = [
  {
    title: 'Enhancement is one-size-fits-all.',
    body: 'Existing methods enhance every image the same way, whether it needs it or not, and rarely check if small but important details — like tubes or wires — are still visible afterward.',
  },
  {
    title: 'One heatmap has to explain everything.',
    body: 'When a model predicts several diseases at once, most tools give one combined heatmap, so there is no way to tell which part of the image actually supports which diagnosis.',
  },
  {
    title: 'AI reports are not fact-checked properly.',
    body: 'Report-writing models can sound confident while being wrong, and most checking methods either verify every line the same way, or do not verify anything at all.',
  },
]

export const NOVELTY = [
  {
    n: '01',
    title: 'Conditional Image Enhancement',
    points: [
      'Quality-aware preprocessing using BRISQUE and entropy',
      'Decision-based pipeline: enhance only if quality is low',
      'Attention U-Net + CLAHE: best average rank of 8 methods compared',
      'Avoids unnecessary enhancement of good images; preserves clinical details such as support devices',
    ],
  },
  {
    n: '02',
    title: 'Multi-Disease Detection with Class-Specific Explainability',
    points: [
      'Four architectures — CNN, Hybrid CNN-Transformer, CheXGCN — under one leak-free protocol',
      'Score-CAM evaluates activation maps by model confidence (no gradients required)',
      'One saliency map per predicted disease in the multi-label setting',
      'Output: disease probabilities with independent visual explanations',
    ],
  },
  {
    n: '03',
    title: 'Trust-Aware Report Generation using VLM + RAG',
    points: [
      'MediVLM turns region-aware visual features into a clinical narrative',
      'RadCheck RAG extracts clinical claims and runs an LLM baseline check',
      'PubMed grounding only for atypical claims; MKG contradiction engine for consistency',
      'Final verdict: hallucinated / not hallucinated with a confidence score',
    ],
  },
]

// Interactive pipeline diagram on the home page
export const STAGES = [
  {
    id: 'input',
    label: 'Chest X-ray',
    sub: 'Frontal view (DICOM / JPEG)',
    color: 'muted',
    details: {
      what: 'A frontal chest radiograph enters the pipeline. Every stage works on a 256×256 grayscale image.',
      bullets: ['Accepts PNG / JPEG / DICOM', 'Quality assessment: BRISQUE, Entropy, Sharpness'],
    },
  },
  {
    id: 'stage1',
    label: 'Stage 1 · Conditional Enhancement',
    sub: 'Quality gate → Attention U-Net + CLAHE',
    color: 'primary',
    route: '/enhance',
    details: {
      what: 'A quality gate decides whether the X-ray needs enhancement. Low-quality images go through the Attention U-Net (ResBlocks, CBAM, attention gates) followed by CLAHE; good images skip enhancement unchanged.',
      bullets: [
        'Residual blocks preserve important features and improve feature flow',
        'CBAM applies channel and spatial attention at the bottleneck',
        'Attention gates pass relevant encoder features through the skip connections',
        'CLAHE boosts local contrast while limiting noise amplification',
      ],
    },
  },
  {
    id: 'stage2',
    label: 'Stage 2 · Detection + Score-CAM',
    sub: '4 architectures · per-disease heatmaps',
    color: 'accent',
    route: '/detect',
    details: {
      what: 'Multi-label classification of Cardiomegaly, Edema and Pleural Effusion with four architectures trained on the same patient-level split. Score-CAM then produces one independent heatmap per disease.',
      bullets: [
        'CNN baseline · CNN matched recipe · Hybrid CNN-Transformer · CheXGCN',
        'Thresholds tuned per disease on validation data only',
        'Score-CAM on denseblock4: gradient-free, each activation map scored by a forward pass',
        'One heatmap per disease, each verifiable on its own',
      ],
    },
  },
  {
    id: 'stage3',
    label: 'Stage 3 · Report + RadCheck',
    sub: 'MediVLM → RadCheck RAG',
    color: 'd-cardio',
    route: '/report',
    details: {
      what: 'MediVLM (Faster R-CNN regions + CLIP ViT-L/14 + GPT-2) writes the radiology report. RadCheck RAG then extracts clinical claims, checks them against an LLM baseline, grounds atypical claims in PubMed and runs a medical-knowledge-graph contradiction engine.',
      bullets: [
        'Region-aware visual tokens: appearance + spatial position',
        'Claims classified as typical / atypical',
        'PubMed grounding only if atypical',
        'Verdict: hallucinated / not hallucinated + confidence',
      ],
    },
  },
  {
    id: 'output',
    label: 'Structured Diagnostic Output',
    sub: 'Image · scores · heatmaps · verified report',
    color: 'ok',
    route: '/pipeline',
    details: {
      what: 'Enhanced image, disease confidence scores, per-disease saliency maps and a RAG-grounded clinical report — all in one structured output.',
      bullets: ['Enhanced X-ray', 'Disease confidence scores', 'Per-disease saliency maps', 'RAG-grounded clinical report'],
    },
  },
]

export const TEAM = [
  { name: 'B. Mohan Sai', reg: 'CB.SC.U4CSE23409', photo: '/team/mohan_sai.jpg', linkedin: 'https://www.linkedin.com/in/mohan-sai-buchireddygari-559987393', github: 'https://github.com/MOHANSAIBUCHIREDDYGARI' },
  { name: 'Ch. Surya Sampreeth', reg: 'CB.SC.U4CSE23410', photo: '/team/surya_sampreeth.jpg', linkedin: 'https://www.linkedin.com/in/surya-sampreeth-chirla-564646321/', github: 'https://github.com/SuryaSampreeth' },
  { name: 'Ch. Sai Rithivik', reg: 'CB.SC.U4CSE23411', photo: '/team/sai_rithivik.jpg', linkedin: 'https://www.linkedin.com/in/chivukula-sai-rithivik-901a79371', github: 'https://github.com/rithivik-1512' },
  { name: 'K. Jayesh Rao', reg: 'CB.SC.U4CSE23418', photo: '/team/jayesh_rao.jpg', linkedin: 'https://www.linkedin.com/in/jayesh-rao-kalla-b3a75929a/', github: 'https://github.com/JAY-cloudbuster' },
  { name: 'T. Srihitha', reg: 'CB.SC.U4CSE23450', photo: '/team/srihitha.jpg', linkedin: 'https://www.linkedin.com/in/srihitha-thotakura-38a5b23b2/', github: 'https://github.com/srihithathotakura' },
]

// Milestones as presented in the two reviews. No dates or percentages are given in the PDFs.
export const TIMELINE = [
  {
    phase: 'Review 1',
    title: 'Change in project direction',
    status: 'done',
    body: 'Moved from cognitive-load estimation on the STEW EEG dataset (no paired EEG + video data, limited scalability) to explainable chest X-ray diagnosis.',
  },
  {
    phase: 'Review 1',
    title: 'Literature survey & novelty',
    status: 'done',
    body: '10 papers reviewed (UniMIE, DiffCode, Score-CAM, FM-G-CAM, CAM survey, UniRG-CXR, AMG-RAG, MediVLM, DenseNet, CheXNet); three novelties defined.',
  },
  {
    phase: 'Review 2',
    title: 'Stage 1 — Image enhancement',
    status: 'done',
    body: '8 methods implemented and compared on 23 low-quality CheXpert images. Attention U-Net + CLAHE selected (best average rank 2.2).',
  },
  {
    phase: 'Review 2',
    title: 'Stage 2 — Detection under one protocol',
    status: 'done',
    body: 'Shared data foundation (patient-level 70/15/15 split); 4 techniques compared at 1k, 6k and 20k images per disease.',
  },
  {
    phase: 'Review 2',
    title: 'Per-disease Score-CAM',
    status: 'done',
    body: 'Independent Score-CAM heatmaps on denseblock4 for all three diseases; CheXGCN gave the clearest per-disease differentiation.',
  },
  {
    phase: 'Review 2',
    title: 'Stage 3 — MediVLM + RadCheck RAG',
    status: 'done',
    body: 'MediVLM reproduced with GPT-2 and Qwen decoders; RadCheck redesigned with LLM baseline check, selective PubMed grounding and MKG contradiction engine.',
  },
  {
    phase: 'Current focus',
    title: 'End-to-end integration',
    status: 'active',
    body: 'Integrating the enhancement module into the complete pipeline: quality gate → enhancement → detection → Score-CAM → report → RadCheck.',
  },
  {
    phase: 'Ongoing',
    title: 'Technical enrichment courses',
    status: 'active',
    body: 'AI for Medical Diagnosis · Generative AI with LLMs · End-to-End Multimodal AI: Fine-Tuning, Fusion & MLOps (Coursera).',
  },
]

// "How to use" — shown on the home page, in the navbar help drawer and in the command palette
export const HOW_TO = {
  steps: [
    { title: 'Pick a stage', body: 'Use the navbar: Enhancement, Detection, Report — or Pipeline to run all three in one go.' },
    { title: 'Add a chest X-ray', body: 'Drag & drop a PNG / JPEG / DICOM (max 10 MB), or click one of the four sample X-rays.' },
    { title: 'Choose a model & run', body: 'Select a model card where offered and press Run. A timer shows progress; real models take seconds to ~1.5 min on CPU.' },
    { title: 'Explore & download', body: 'Compare with the slider, toggle per-disease heatmaps, read the verified report, then download PNG / ZIP / PDF.' },
  ],
  pages: {
    '/enhance': [
      'Drag the slider handle (or use ← →) to compare original and enhanced.',
      '"Quality gate" mode enhances only low-quality images — the core idea of Stage 1.',
      'Metrics below the image show BRISQUE ↓, Entropy ↑, Sharpness ↑, CII ↑ and EME ↑ for your X-ray.',
    ],
    '/detect': [
      'Click a disease bar or chip to show that disease’s own Score-CAM heatmap.',
      'Use the opacity slider to blend the heatmap; "Side by side" and "All diseases" change the layout.',
      'The tick on each bar is the decision threshold — above it the disease is predicted positive.',
    ],
    '/report': [
      'Generate a report, then read the RadCheck verdict and confidence gauge.',
      'Underlined sentences are verified claims; yellow = unsupported, red = contradicted.',
      'Use Copy or PDF to export the report.',
    ],
    '/pipeline': [
      'Choose the Stage 1 and Stage 2 models, then press Run — each stage’s output appears as soon as it finishes.',
      'The summary dashboard at the top collects every result once the run completes.',
      '"Download all (ZIP)" saves images, heatmaps, JSON and the PDF report.',
    ],
    '/': [
      'Use the section bar under the navbar (or Ctrl + K) to jump to any section.',
      'Click a stage in the Methodology diagram to see how it works.',
      'Press "Try Demo" to run the complete pipeline on a sample X-ray.',
    ],
  },
}

// Home-page sections, in page order (section bar + command palette)
export const HOME_SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'how-to-use', label: 'How to use' },
  { id: 'objectives', label: 'Objectives' },
  { id: 'problem', label: 'Problem' },
  { id: 'methodology', label: 'Methodology' },
  { id: 'results', label: 'Results' },
  { id: 'novelty', label: 'Novelty' },
  { id: 'team', label: 'Team' },
  { id: 'progress', label: 'Progress' },
  { id: 'docs', label: 'Docs' },
  { id: 'github', label: 'GitHub' },
  { id: 'video', label: 'Video' },
]

export const DATASETS = [
  {
    name: 'CheXpert',
    use: 'Enhancement + disease detection',
    stats: [
      { value: 224316, label: 'X-ray images' },
      { value: 65240, label: 'Patients' },
      { value: 14, label: 'Observations' },
    ],
    body: 'Large-scale chest X-ray dataset from Stanford with uncertainty labels. Frontal views only; balanced per-disease subsample; patient-level 70/15/15 split with zero leakage.',
  },
  {
    name: 'MIMIC-CXR',
    use: 'Report generation (MediVLM)',
    stats: [
      { value: 369, suffix: 'K', label: 'Train' },
      { value: 3.0, suffix: 'K', decimals: 1, label: 'Val' },
      { value: 5.2, suffix: 'K', decimals: 1, label: 'Test' },
    ],
    body: 'Large-scale chest X-rays paired with free-text radiology reports. Training: AdamW, LR 2×10⁻⁵, batch 32, 30–50 epochs.',
  },
]

// TEAM_95_PR2 slide 9 — mean over 23 low-quality CheXpert images (lower BRISQUE / higher others = better)
export const ENHANCEMENT_LEADERBOARD = [
  { method: 'Attn U-Net + CLAHE', brisque: 13.35, entropy: 7.92, sharpness: 2082.3, cii: 1.9, rank: 2.2, ours: true },
  { method: 'U-Net V3 + CLAHE', brisque: 17.51, entropy: 7.91, sharpness: 1859.7, cii: 1.76, rank: 3.2 },
  { method: 'CLAHE', brisque: 7.01, entropy: 7.84, sharpness: 3609.6, cii: 1.71, rank: 3.8 },
  { method: 'Attention U-Net', brisque: 13.22, entropy: 7.86, sharpness: 1295.7, cii: 1.37, rank: 4.6 },
  { method: 'Original', brisque: 5.62, entropy: 7.85, sharpness: 534.3, cii: 1.0, rank: 5.2 },
  { method: 'Zero-DCE', brisque: 19.59, entropy: 6.54, sharpness: 1766.4, cii: 1.27, rank: 5.4 },
  { method: 'U-Net V3', brisque: 17.96, entropy: 7.86, sharpness: 1190.9, cii: 1.29, rank: 5.6 },
  { method: 'UniMIE', brisque: -0.73, entropy: 7.85, sharpness: 725.5, cii: 0.93, rank: 6.0 },
]

export const ENHANCEMENT_HIGHLIGHTS = [
  { value: 2.2, decimals: 1, label: 'Best average rank', sub: 'across BRISQUE, Entropy, Sharpness, CII & EME' },
  { value: 7.92, decimals: 2, label: 'Highest entropy', sub: 'retains the most diagnostically-relevant detail' },
  { value: 1.9, decimals: 2, label: 'Highest CII', sub: 'best contrast improvement of all 8 methods' },
]

export const DISEASES = ['Cardiomegaly', 'Edema', 'Pleural Effusion']

// Held-out test set, 6k images per disease (detection notebook; matches config.yaml)
export const DETECTION_RESULTS = [
  { model: 'CNN (baseline)', auc: [0.832, 0.803, 0.841], f1: [0.668, 0.719, 0.776] },
  { model: 'CNN (matched)', auc: [0.826, 0.801, 0.839], f1: [0.654, 0.72, 0.782] },
  { model: 'Hybrid', auc: [0.835, 0.79, 0.845], f1: [0.674, 0.706, 0.785] },
  { model: 'CheXGCN', auc: [0.829, 0.804, 0.842], f1: [0.667, 0.721, 0.773] },
]

// TEAM_95_PR2 slide 17 — AUC rank per disease x data scale (1 = best)
export const AUC_RANKS = {
  columns: ['Card. 1k', 'Edema 1k', 'Eff. 1k', 'Card. 6k', 'Edema 6k', 'Eff. 6k', 'Card. 20k', 'Edema 20k', 'Eff. 20k'],
  rows: [
    { model: 'CNN (baseline)', ranks: [1, 1, 1, 2, 2, 3, 1, 2, 1] },
    { model: 'CNN (matched)', ranks: [2, 2, 2, 4, 3, 4, 4, 1, 4] },
    { model: 'Hybrid', ranks: [3, 3, 3, 1, 4, 1, 2, 4, 3] },
    { model: 'CheXGCN', ranks: [4, 4, 4, 3, 1, 2, 3, 3, 2] },
  ],
}

export const DETECTION_TAKEAWAYS = [
  'CheXGCN improves the most with more data: lowest AUC at 1k, tied for highest on Cardiomegaly and Pleural Effusion at 20k.',
  'Most gains happen between 1k and 6k images; 6k → 20k plateaus, showing diminishing returns.',
  'The Hybrid CNN-Transformer needs more data to compete — lowest or near-lowest F1/AUC at 1k.',
  'At 6k and 20k no single architecture leads consistently: data volume matters more than architecture at this scale.',
]

// TEAM_95_PR2 slide 25
export const REPORT_METRICS = [
  { metric: 'BLEU-1', gpt2: 0.3631, qwen: 0.2111 },
  { metric: 'BLEU-2', gpt2: 0.2401, qwen: 0.1444 },
  { metric: 'BLEU-3', gpt2: 0.1707, qwen: 0.1061 },
  { metric: 'BLEU-4', gpt2: 0.1228, qwen: 0.0784 },
  { metric: 'METEOR', gpt2: 0.37, qwen: 0.3456 },
  { metric: 'ROUGE-L', gpt2: 0.27, qwen: 0.2237 },
]

// TEAM_95_PR2 RadCheck slide 6 (System A = proposed architecture, System B = previous architecture)
export const RADCHECK_COMPARISON = [
  { metric: 'Accuracy', proposed: 86.96, previous: 73.91 },
  { metric: 'Precision', proposed: 91.67, previous: 76.92 },
  { metric: 'Recall', proposed: 84.62, previous: 76.92 },
  { metric: 'F1-score', proposed: 88.0, previous: 76.92 },
  { metric: 'Specificity', proposed: 90.0, previous: 70.0 },
]

export const FIGURES = {
  architecture: { src: '/figures/architecture.png', caption: 'ReXTrust architecture (TEAM_95_PR2, slide 5)' },
  attentionUnet: { src: '/figures/attention_unet.png', caption: 'Proposed Attention U-Net + CLAHE building blocks' },
  qualitative: { src: '/figures/enhancement_qualitative.jpg', caption: 'Qualitative comparison — all methods, same input image' },
  featureMaps: { src: '/figures/enhancement_feature_maps.jpg', caption: 'Layer-wise intermediate feature maps — Attention U-Net' },
  metrics: { src: '/figures/enhancement_metrics.png', caption: 'Mean ± std per metric, all 8 methods (23 images)' },
  detail: { src: '/figures/enhancement_detail_preserved.jpg', caption: 'Original vs CLAHE vs Attention U-Net vs Attention U-Net + CLAHE: the support device stays visible' },
  aucRanks: { src: '/figures/auc_rank_grid.png', caption: 'AUC rank per disease × data scale' },
  aucScales: { src: '/figures/auc_across_scales.png', caption: 'ROC-AUC across data scales — all 4 techniques (test set)' },
  f1Scales: { src: '/figures/f1_across_scales.png', caption: 'F1 across data scales — all 4 techniques (test set)' },
  scorecam: { src: '/figures/scorecam_chexgcn.jpg', caption: 'CheXGCN — independent per-disease Score-CAM heatmaps' },
  radcheck: { src: '/figures/radcheck_comparison.png', caption: 'RadCheck: proposed (System A) vs previous (System B) architecture' },
  chexpert: { src: '/figures/chexpert_distribution.png', caption: 'CheXpert class distribution' },
}

export const DOCUMENTS = [
  { title: 'Project Review 2 — Full deck', file: '/pdf/TEAM_95_PR2.pdf', pages: 35, desc: 'Enhancement, detection, Score-CAM, MediVLM and RadCheck RAG — methods and results.' },
  { title: 'Review 2 — Literature & novelty', file: '/pdf/Review_2.pdf', pages: 17, desc: 'Change in direction, literature review, novelties, architecture and dataset.' },
]

export const REFERENCES = [
  'B. Fei et al., "A diffusion model for universal medical image enhancement," Communications Medicine, vol. 5, p. 294, 2025.',
  'Zero-DCE: Guo et al., IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.',
  'U-Net: Ronneberger et al., Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.',
  'J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," AAAI, 2019.',
  'H. Wang et al., "Score-CAM: Score-weighted visual explanations for convolutional neural networks," CVPRW, 2020.',
  'D. Goswami, R. Subedi, S. Chakraborty, "MediVLM: A vision language model for radiology report generation from medical images," Findings of EMNLP, 2025.',
  'Agentic Medical Knowledge Graphs Enhance Medical Question Answering (AMG-RAG).',
]

export const DISCLAIMER = 'For research and educational purposes only. Not for clinical use.'

// ─── Date Formatting Utilities ─────────────────────────────────
export const formatRelativeTime = (dateStr) => {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return diffMins + 'm ago';
  if (diffHours < 24) return diffHours + 'h ago';
  if (diffDays < 7) return diffDays + 'd ago';
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
};
