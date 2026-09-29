# Extracted unchanged from full_pipeline.ipynb (RadCheck cell) so the Gradio app can import it.
"""
AMG-RAG: Clinical Report Hallucination Detection System
5-Step Pipeline:
  1. Report Ingestion & Extraction
  2. LLM Baseline Check (typical / rare / impossible)
  3. PubMed Grounding — only for rare findings, with an LLM judging whether the abstracts support the claim
  4. MKG Contradiction Engine
  5. Final Verdict: Hallucinated / Consistent / Undetermined with confidence score

Note: this checks the report text for plausibility and internal consistency only. It never sees the
image, so a fluent, self-consistent report that is wrong about the image will still pass.
"""

import time
import os
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
from langchain_core.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser  # pyright: ignore[reportMissingImports]
import requests
from xml.etree import ElementTree as ET
# Configuration (keys come from the environment / .env, loaded by pipeline_core)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
PUBMED_API_KEY = os.environ.get('pubmed_api') or None


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────
@dataclass
class MedicalEntity:
    """Represents a medical entity in the knowledge graph"""
    name: str
    description: str
    entity_type: str
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)

@dataclass
class MedicalRelation:
    """Represents a relationship between medical entities"""
    source: str
    target: str
    relation_type: str
    confidence: float
    evidence: str
    sources: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Medical Knowledge Graph
# ──────────────────────────────────────────────
class MedicalKnowledgeGraph:
    """Dynamic Medical Knowledge Graph with confidence scoring"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relations = []

    def add_entity(self, entity: MedicalEntity):
        self.entities[entity.name] = entity
        self.graph.add_node(
            entity.name,
            description=entity.description,
            entity_type=entity.entity_type,
            confidence=entity.confidence,
            sources=entity.sources,
        )

    def add_relation(self, relation: MedicalRelation):
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            evidence=relation.evidence,
            sources=relation.sources,
        )

    def get_connected_nodes(self, node_name: str, confidence_threshold: float = 0.0):
        connected = []
        if node_name in self.graph:
            for neighbor in self.graph.neighbors(node_name):
                edge_data = self.graph[node_name][neighbor]
                if edge_data.get("confidence", 0) >= confidence_threshold:
                    connected.append(
                        {
                            "node": neighbor,
                            "relation": edge_data.get("relation_type"),
                            "confidence": edge_data.get("confidence"),
                            "evidence": edge_data.get("evidence"),
                        }
                    )
        return connected


# ──────────────────────────────────────────────
# PubMed Searcher
# ──────────────────────────────────────────────
class PubMedSearcher:
    """PubMed API wrapper for medical literature search"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, max_results: int = 3) -> List[str]:
        """Returns one 'title. abstract' string per article."""
        time.sleep(0.5)
        search_url = f"{self.base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": max_results,
        }
        if self.api_key:
            search_params["api_key"] = self.api_key

        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            if response.status_code != 200:
                return []
            if not response.text.strip().startswith("<"):
                return []

            root = ET.fromstring(response.text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            if not pmids:
                return []

            time.sleep(0.5)
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key

            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            if response.status_code != 200:
                return []

            # Parse per article from XML; the plain-text format splits on paragraphs, not articles,
            # which returned title/journal/affiliation fragments instead of abstracts.
            root = ET.fromstring(response.text)
            abstracts = []
            for article in root.findall(".//PubmedArticle"):
                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""
                body = " ".join("".join(a.itertext()) for a in article.findall(".//AbstractText"))
                if body.strip():
                    abstracts.append(f"{title.strip()} {body.strip()}".strip())
            return abstracts

        except Exception as e:
            print(f"  PubMed search error: {e}")
            return []


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _align_lists(*lists, fill="N/A"):
    """Force all extracted arrays to the same length so index-based
    alignment (findings[i] <-> anatomical_targets[i] <-> ...) can't
    silently drift when the LLM returns mismatched array lengths."""
    max_len = max((len(l) for l in lists), default=0)
    return [list(l) + [fill] * (max_len - len(l)) for l in lists]


def _short_error(e) -> str:
    """One readable line instead of the API's full JSON error payload."""
    s = str(e)
    if "RESOURCE_EXHAUSTED" in s or " 429" in s:
        m = re.search(r"retry in ((?:\d+h)?(?:\d+m)?)", s)
        return "Gemini API quota exhausted" + (f", resets in {m.group(1)}" if m and m.group(1) else "")
    if "DEADLINE_EXCEEDED" in s or " 504" in s:
        return "Gemini API timed out (504)"
    if "API key" in s or "PERMISSION_DENIED" in s or " 403" in s:
        return "Gemini API key rejected"
    return s.strip().splitlines()[0][:160] if s.strip() else type(e).__name__


def _as_list(value):
    """LLM fields sometimes come back as a bare string/number (or None) instead of a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_str_list(value):
    return [v if isinstance(v, str) else json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for v in _as_list(value)]


def _as_float(value, default=0.5):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_contradictions(descriptions, raw_pairs, severities, num_findings):
    """The LLM is asked for pairs like [0, 2] but the scoring logic needs
    a list of index-lists, e.g. [[0, 2], [1, 3]]. Handle both shapes and
    drop anything that doesn't parse as a valid pair of in-range indices.

    Descriptions and severities are validated together with their pair, so dropping an invalid
    pair can't shift later descriptions/severities onto the wrong pair. A missing description or
    severity gets a default instead of discarding the pair."""
    if not raw_pairs:
        return [], [], []

    # Case: flat list like [0, 2, 1, 3] -> treat as consecutive pairs
    if all(isinstance(x, (int, float)) for x in raw_pairs):
        it = iter(raw_pairs)
        raw_pairs = [[a, b] for a, b in zip(it, it)]

    out_desc, out_pairs, out_sev = [], [], []
    for k, pair in enumerate(raw_pairs):
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            continue
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if not (0 <= a < num_findings and 0 <= b < num_findings):
            continue
        out_pairs.append([a, b])
        out_desc.append(descriptions[k] if k < len(descriptions) else "Contradiction detected")
        out_sev.append(severities[k] if k < len(severities) else 0.5)
    return out_desc, out_pairs, out_sev


def _normalize_class(label: str) -> str:
    """Map the LLM's label onto typical / rare / impossible (tolerates the old 'atypical' wording)."""
    c = label.strip().lower()
    if "impossib" in c or "contradict" in c:
        return "impossible"
    if "rare" in c or "atypical" in c:
        return "rare"
    return "typical"


def _split_sentences(text: str) -> List[str]:
    """Split on sentence-ending periods only, so decimals like '1.5 cm' stay intact."""
    return [s.strip(" .") for s in re.split(r"(?<=\.)\s+|\n+", text) if s.strip(" .")]


# ──────────────────────────────────────────────
# Main System
# ──────────────────────────────────────────────
class AMG_RAG_ReportSystem:
    """AMG-RAG system for Clinical Report Hallucination Detection"""

    def __init__(self, google_api_key: str = None):
        if google_api_key:
            # gemini-2.5-flash answers in ~2 s; gemma-4-31b-it kept hitting 504 timeouts (~10 min per report).
            # Each Gemini model has its own free-tier quota (gemini-2.5-flash: 20 requests/day), so when one is
            # exhausted or times out the next model in the chain answers instead.
            # Override with RADCHECK_MODEL and RADCHECK_FALLBACK_MODELS (comma-separated).
            self.models = [os.environ.get("RADCHECK_MODEL", "gemini-2.5-flash")] + [
                m.strip() for m in os.environ.get("RADCHECK_FALLBACK_MODELS",
                                                  "gemini-3.5-flash,gemini-2.5-flash-lite").split(",") if m.strip()]
            llms = [ChatGoogleGenerativeAI(
                model=m,
                temperature=0.0,
                timeout=120,      # added: fail instead of hanging
                max_retries=1,    # added
                google_api_key=google_api_key,
            ) for m in self.models]
            self.llm = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]
        else:
            raise ValueError("Google API key is required.")

        self.kg = MedicalKnowledgeGraph()
        self.pubmed = PubMedSearcher(api_key=PUBMED_API_KEY)
        self._setup_chains()

    # ──────────────────────────────────────────
    # Chain Setup
    # ──────────────────────────────────────────
    def _setup_chains(self):
        """Setup LLM chains for the 5-step report pipeline"""

        # ── Step 1 chain: Report Ingestion & Extraction ──
        finding_schemas = [
            ResponseSchema(
                name="findings",
                description="List of clinical claims/findings parsed from the report",
                type="array",
            ),
            ResponseSchema(
                name="anatomical_targets",
                description="The anatomical structures associated with each finding",
                type="array",
            ),
            ResponseSchema(
                name="clinical_status",
                description="Clinical observation status for each finding (e.g. 'normal', 'abnormal', 'clear')",
                type="array",
            ),
        ]
        finding_parser = StructuredOutputParser.from_response_schemas(finding_schemas)

        self.finding_extractor = (
            PromptTemplate(
                template="""Extract all individual medical findings/claims and associated anatomical structures from this clinical report.

            Report: {report}

            For each finding/claim, provide:
            1. The finding statement
            2. The anatomical target structure
            3. The clinical status (e.g. normal, abnormal, clear, consolidated, etc.)

            IMPORTANT: The three output arrays (findings, anatomical_targets, clinical_status)
            MUST all be the same length and index-aligned — element i of each array must
            describe the same finding.

            Return in JSON format:
            {format_instructions}""",
                input_variables=["report"],
                partial_variables={
                    "format_instructions": finding_parser.get_format_instructions()
                },
            )
            | self.llm
            | finding_parser
        )

        # ── Step 2 chain: LLM Baseline Check ──
        baseline_schemas = [
            ResponseSchema(
                name="classifications",
                description="For each finding: 'typical', 'rare', or 'impossible' (see definitions)",
                type="array",
            ),
            ResponseSchema(
                name="reasoning",
                description="Brief reasoning for each classification",
                type="array",
            ),
        ]
        baseline_parser = StructuredOutputParser.from_response_schemas(
            baseline_schemas
        )

        self.baseline_checker = (
            PromptTemplate(
                template="""You are a clinical radiology expert. For each finding below, classify it as exactly one of 'typical', 'rare', or 'impossible'.

            DEFINITIONS:
            - 'typical': A standard, commonly seen clinical observation (e.g. 'heart is normal size', 'lungs are clear', 'no pleural effusion', 'mild cardiomegaly', 'pneumothorax').
            - 'rare': A real but uncommon or unusual finding that needs literature support to be believed
              (e.g. extremely rare congenital conditions such as 'dextrocardia', 'situs inversus').
            - 'impossible': A finding that cannot be true as written. This includes:
              a) Internally contradictory statements (e.g. 'normal size with severe cardiomegaly', 'clear lungs with massive consolidation')
              b) Impossible whole-organ counts (e.g. 'patient has 3 lungs')
              c) Impossible or incorrect counts of normal anatomical SUBSTRUCTURES — lobes, chambers,
                 valves, vessels, etc. — relative to known human anatomy. For example: the left lung
                 normally has 2 lobes and the right lung has 3 lobes, so "three lobes in the left lung"
                 is impossible. The heart normally has 4 chambers, so "five chambers" is impossible.
                 Apply this same substructure-count check to any anatomical structure mentioned,
                 not just the examples given here.

            NOTE: Common abnormal findings (e.g. pleural effusion, tumors) are still 'typical'. Rare-but-real
            conditions are 'rare', NOT 'impossible'.

            Findings to classify:
            {findings_list}

            IMPORTANT: Return exactly one classification and one reasoning entry per finding listed above, in the same order.

            Return in JSON format:
            {format_instructions}""",
                input_variables=["findings_list"],
                partial_variables={
                    "format_instructions": baseline_parser.get_format_instructions()
                },
            )
            | self.llm
            | baseline_parser
        )

        # ── Step 3 chain: judge whether retrieved PubMed abstracts support a rare finding ──
        evidence_schemas = [
            ResponseSchema(
                name="verdict",
                description="'supported' if the abstracts show this finding is real and can present this way, "
                            "'refuted' if they show it cannot be true, 'not_addressed' if they are off-topic or inconclusive",
            ),
            ResponseSchema(name="reasoning", description="One-sentence justification"),
        ]
        evidence_parser = StructuredOutputParser.from_response_schemas(evidence_schemas)

        self.evidence_judge = (
            PromptTemplate(
                template="""You are checking whether medical literature supports a finding from a radiology report.
            Retrieval was by keyword, so the abstracts may be off-topic. Only answer 'supported' if the
            abstracts actually describe this finding as a real clinical presentation.

            Finding: {finding}

            Abstracts:
            {abstracts}

            Return in JSON format:
            {format_instructions}""",
                input_variables=["finding", "abstracts"],
                partial_variables={"format_instructions": evidence_parser.get_format_instructions()},
            )
            | self.llm
            | evidence_parser
        )

        # ── Step 4 chain: MKG Contradiction Engine ──
        contradiction_schemas = [
            ResponseSchema(
                name="contradictions",
                description="List of contradiction descriptions found between findings. Empty list if none found.",
                type="array",
            ),
            ResponseSchema(
                name="contradiction_pairs",
                description="List of [index_a, index_b] pairs indicating which findings are involved in each contradiction. For a contradiction between two different findings, use their two distinct indices, e.g. [0, 2]. For a contradiction involving only ONE finding (self-contradictory statement, or physiologically/anatomically impossible claim), use the SAME index twice, e.g. [5, 5]. Each element must be a 2-item list. Empty list if none.",
                type="array",
            ),
            ResponseSchema(
                name="severity_scores",
                description="Severity score (0.0 to 1.0) for each contradiction, same order/length as contradictions. 1.0 = definite contradiction, 0.5 = possible, 0.0 = no contradiction.",
                type="array",
            ),
        ]
        contradiction_parser = StructuredOutputParser.from_response_schemas(
            contradiction_schemas
        )

        self.contradiction_detector = (
            PromptTemplate(
                template="""You are a clinical contradiction detection engine. Analyze the following clinical findings from a single medical report for internal contradictions.

            A CONTRADICTION occurs when:
            1. Two DIFFERENT findings about the SAME anatomical structure make opposite claims (e.g. 'heart is normal size' AND 'severe cardiomegaly'). Encode as [index_a, index_b] with two distinct indices.
            2. A single finding contains self-contradictory statements (e.g. 'clear lungs with massive consolidation'). Encode as [index, index] — the SAME index twice, since only one finding is involved.
            3. A finding describes something physiologically or anatomically impossible (e.g. 'patient has 3 lungs', 'three lobes in the left lung', 'five cardiac chambers'). Encode as [index, index] — the SAME index twice, since only one finding is involved.

            NOT a contradiction (these are normal, expected patterns in clinical radiology):
            1. Multiple abnormal findings in different regions (e.g. effusion in lungs + cardiomegaly in heart)
            2. A finding described as changed relative to a prior study
            3. Minor/trace findings coexisting with general "clear" or "no acute findings" statements
               - In radiology convention, "clear lungs" means no consolidation/infiltrate/pneumonia, NOT absence of ALL findings
               - Minor findings like mild basilar atelectasis, trace granulomas, or small nodules can coexist with "clear" statements
               - This is standard practice and NOT a contradiction (e.g. "lungs are clear" + "mild basilar atelectasis" = NORMAL, not contradictory)
            4. Incidental findings mentioned alongside primary pathology (e.g. "no pneumonia" + "small granuloma noted")
            5. ACUITY MISMATCH RULE: "No acute findings" is a claim about the absence of new/emergent
                  pathology — it is NOT a claim that the study is entirely normal, and it does NOT
                  contradict a finding unless that finding is itself explicitly framed as acute, new,
                  or an interval change.

                  To evaluate whether a pairing is a contradiction, classify the abnormal finding's
                  acuity based on its own wording:
                  - Explicitly acute/new/interval-change language ("new," "acute," "increased from prior,"
                    "worsening," "interval development of") → CAN contradict "no acute findings"
                  - No acuity language, or language suggesting a chronic/structural/incidental process
                    ("enlarged," "prominent," "tortuous," "calcified," "stable," "chronic," "old,"
                    "longstanding") → does NOT contradict "no acute findings" on its own

                  Do not infer acuity from the finding's anatomical category or clinical severity —
                  infer it only from the acuity language actually present in that finding's text.
                  If acuity is ambiguous or unstated, do NOT flag it as a contradiction
                  (default to non-contradiction when acuity is unclear).

                  Examples (illustrative only, not exhaustive): enlarged pulmonary arteries, old rib
                  fracture, stable pulmonary nodule, chronic scarring, calcified granuloma.

            Findings:
            {findings_list}

            Knowledge Graph Context:
            {kg_context}

            Identify ONLY true logical contradictions. Each entry in contradiction_pairs MUST be
            a 2-item list of finding numbers exactly as labelled above (Finding 0, Finding 1, ...) — [index_a, index_b] for a two-finding
            contradiction (type 1), or [index, index] for a single-finding contradiction
            (types 2 and 3). contradictions / contradiction_pairs / severity_scores MUST all be
            the same length, in matching order.

            Return in JSON format:
            {format_instructions}""",
                input_variables=["findings_list", "kg_context"],
                partial_variables={
                    "format_instructions": contradiction_parser.get_format_instructions()
                },
            )
            | self.llm
            | contradiction_parser
        )

    # ──────────────────────────────────────────
    # Main Pipeline
    # ──────────────────────────────────────────
    def evaluate_medical_report(self, report: str) -> Dict[str, Any]:
        """Execute the 5-step report hallucination detection pipeline."""
        # Steps that fell back to a default because the LLM call failed or returned too little.
        # A non-empty list means the verdict is not fully verified.
        llm_errors = []

        # ════════════════════════════════════════
        # STEP 1: Report Ingestion & Extraction
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 1: Report Ingestion & Extraction")
        print("=" * 60)
        print(f"\n  Input Report:\n  \"{report}\"\n")

        try:
            finding_result = self.finding_extractor.invoke({"report": report})
            findings = _as_str_list(finding_result.get("findings"))
            anatomical_targets = _as_str_list(finding_result.get("anatomical_targets"))
            clinical_statuses = _as_str_list(finding_result.get("clinical_status"))
            if not findings:
                raise ValueError("LLM returned no findings")
        except Exception as e:
            print(f"  Finding extraction error: {e}")
            llm_errors.append(f"Step 1 extraction failed ({_short_error(e)}); report split on sentences instead")
            plain = report.replace("FINDINGS:", " ").replace("IMPRESSION:", " ")  # section labels aren't claims
            findings = _split_sentences(plain)
            anatomical_targets = ["General"] * len(findings)
            clinical_statuses = ["Unspecified"] * len(findings)

        # FIXED: enforce index alignment instead of relying on ad-hoc
        # "if i < len(...)" guards scattered through the rest of the pipeline.
        findings, anatomical_targets, clinical_statuses = _align_lists(
            findings, anatomical_targets, clinical_statuses
        )

        print(f"  Extracted {len(findings)} clinical claims:\n")
        for i, f in enumerate(findings):
            print(f"    Claim {i+1}: \"{f}\"")
            print(f"             Target: {anatomical_targets[i]} | Status: {clinical_statuses[i]}")

        # ════════════════════════════════════════
        # STEP 2: LLM Baseline Check
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 2: LLM Baseline Check (typical / rare / impossible)")
        print("=" * 60)

        # Numbered from 0 so the indices the contradiction engine returns map straight onto findings[i].
        findings_formatted = "\n".join(
            [
                f"  Finding {i}: \"{findings[i]}\" (Target: {anatomical_targets[i]}, Status: {clinical_statuses[i]})"
                for i in range(len(findings))
            ]
        )

        try:
            baseline_result = self.baseline_checker.invoke(
                {"findings_list": findings_formatted}
            )
            classifications = _as_str_list(baseline_result.get("classifications"))
            baseline_reasoning = _as_str_list(baseline_result.get("reasoning"))
        except Exception as e:
            print(f"  Baseline check error: {e}")
            llm_errors.append(f"Step 2 baseline check failed ({_short_error(e)})")
            classifications = []
            baseline_reasoning = []

        # FIXED: pad/align instead of silently defaulting everything to
        # "typical" on a partial or failed response.
        # Align to the number of findings, not just to each other: Gemma sometimes returns fewer
        # classifications than findings, and indexing past the end raised "list index out of range".
        n_findings = len(findings)
        if 0 < len(classifications) < n_findings:
            llm_errors.append(f"Step 2 classified only {len(classifications)} of {n_findings} findings; "
                              f"the rest default to 'typical'")
        classifications = [_normalize_class(c) for c in classifications[:n_findings]]
        classifications += ["typical"] * (n_findings - len(classifications))
        baseline_reasoning = baseline_reasoning[:n_findings]
        baseline_reasoning += ["(no classification returned by the LLM)"] * (n_findings - len(baseline_reasoning))

        markers = {"typical": "[OK] TYPICAL", "rare": "[?] RARE", "impossible": "[!] IMPOSSIBLE"}
        rare_indices = []
        for i in range(len(findings)):
            cls = classifications[i]
            print(f"\n    Claim {i+1}: {markers[cls]}")
            print(f"      \"{findings[i]}\"")
            print(f"      Reasoning: {baseline_reasoning[i]}")
            if cls == "rare":
                rare_indices.append(i)

        print(f"\n  Summary: {len(rare_indices)} rare, {classifications.count('impossible')} impossible "
              f"/ {len(findings)} total findings")

        # ════════════════════════════════════════
        # STEP 3: PubMed Grounding (only for rare findings)
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 3: PubMed Grounding — only for rare findings")
        print("=" * 60)

        # index -> 'supported' / 'refuted' / 'not_addressed' / 'no_articles' / 'judge_failed'.
        # Impossible findings skip this step: literature can't make an impossible claim true.
        evidence_verdicts = {}
        if not rare_indices:
            print("\n  No rare findings detected. Skipping PubMed grounding.")
        else:
            for idx in rare_indices:
                finding_text = findings[idx]
                target = anatomical_targets[idx]
                query = f"{target} {finding_text}"
                print(f"\n    Searching PubMed for Claim {idx+1}: \"{finding_text[:80]}...\"")
                abstracts = self.pubmed.search(query, max_results=2)
                if not abstracts:
                    evidence_verdicts[idx] = "no_articles"
                    print("      No PubMed articles found — finding remains ungrounded")
                    continue
                print(f"      Found {len(abstracts)} article(s); judging whether they support the claim")
                try:
                    judged = self.evidence_judge.invoke({
                        "finding": finding_text,
                        "abstracts": "\n\n".join(f"[{j+1}] {a[:1500]}" for j, a in enumerate(abstracts)),
                    })
                    verdict = str(judged.get("verdict", "")).strip().lower()
                    if verdict not in ("supported", "refuted"):
                        verdict = "not_addressed"
                    print(f"      Evidence verdict: {verdict} — {judged.get('reasoning', '')}")
                except Exception as e:
                    print(f"      Evidence judge error: {e}")
                    llm_errors.append(f"Step 3 evidence check failed for claim {idx+1} ({_short_error(e)})")
                    verdict = "judge_failed"
                evidence_verdicts[idx] = verdict

        # ════════════════════════════════════════
        # STEP 4: MKG Contradiction Engine
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 4: MKG Contradiction Engine")
        print("=" * 60)

        # Build Knowledge Graph from findings
        self.kg = MedicalKnowledgeGraph()

        print("\n  Building Medical Knowledge Graph from report claims...")
        for i, target in enumerate(anatomical_targets):
            entity_name = target.title()
            if entity_name not in self.kg.entities:
                med_entity = MedicalEntity(
                    name=entity_name,
                    description=f"Anatomical region: {entity_name}. Status: {clinical_statuses[i]}",
                    entity_type="anatomical_structure",
                    confidence=1.0,
                    sources=["Report Parser"],
                )
                self.kg.add_entity(med_entity)

            finding_name = f"Claim_{i+1}"
            finding_entity = MedicalEntity(
                name=finding_name,
                description=findings[i],
                entity_type="clinical_finding",
                confidence=1.0,
                sources=["Report Parser"],
            )
            self.kg.add_entity(finding_entity)

            relation = MedicalRelation(
                source=entity_name,
                target=finding_name,
                relation_type="has_finding",
                confidence=1.0,
                evidence=f"Status: {clinical_statuses[i]}",
                sources=["Report Parser"],
            )
            self.kg.add_relation(relation)

        print(f"    Nodes: {len(self.kg.entities)} | Edges: {len(self.kg.relations)}")

        # Print KG structure
        for entity_name in self.kg.entities.keys():
            connections = self.kg.get_connected_nodes(entity_name)
            if connections:
                for conn in connections:
                    print(f"    {entity_name} --[{conn['relation']}]--> {conn['node']}")

        # Run contradiction detection
        kg_context_lines = []
        for entity_name in self.kg.entities.keys():
            connections = self.kg.get_connected_nodes(entity_name)
            if connections:
                for conn in connections:
                    kg_context_lines.append(
                        f"{entity_name} --[{conn['relation']}]--> {conn['node']}: {self.kg.entities.get(conn['node'], MedicalEntity('','','',0,[])).description}"
                    )
        kg_context = "\n".join(kg_context_lines) if kg_context_lines else "No graph context available."

        print("\n  Running contradiction detection...")
        try:
            contradiction_result = self.contradiction_detector.invoke(
                {"findings_list": findings_formatted, "kg_context": kg_context}
            )
            contradictions = _as_str_list(contradiction_result.get("contradictions"))
            raw_pairs = _as_list(contradiction_result.get("contradiction_pairs"))
            severity_scores = [_as_float(s) for s in _as_list(contradiction_result.get("severity_scores"))]
        except Exception as e:
            print(f"  Contradiction detection error: {e}")
            llm_errors.append(f"Step 4 contradiction check failed ({_short_error(e)})")
            contradictions = []
            raw_pairs = []
            severity_scores = []

        contradictions, contradiction_pairs, severity_scores = _normalize_contradictions(
            contradictions, raw_pairs, severity_scores, len(findings)
        )

        if contradictions:
            print(f"\n  [!] {len(contradictions)} contradiction(s) detected:\n")
            for i, contradiction in enumerate(contradictions):
                pair = contradiction_pairs[i]
                severity = severity_scores[i]
                print(f"    Contradiction {i+1}: {contradiction}")
                print(f"      Between claims: {pair} | Severity: {severity}")
        else:
            print("\n  [OK] No internal contradictions detected.")

        # ════════════════════════════════════════
        # STEP 5: Final Verdict
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 5: Final Verdict — Hallucinated / Not Hallucinated")
        print("=" * 60)

        # Build per-finding verdict
        detailed_findings = []
        for i in range(len(findings)):
            cls = classifications[i]
            reason = baseline_reasoning[i]

            # Start with baseline confidence
            evidence = evidence_verdicts.get(i)
            if cls == "typical":
                grounding_score = 1.0
            elif cls == "impossible":
                grounding_score = 0.0
            elif evidence == "supported":
                grounding_score = 0.7  # rare, and the literature backs it
            elif evidence == "refuted":
                grounding_score = 0.0
            else:
                grounding_score = 0.3  # rare and ungrounded

            # Check if this finding is involved in a contradiction
            is_contradicted = False
            contradiction_details = []
            for ci, pair in enumerate(contradiction_pairs):
                if i in pair:
                    is_contradicted = True
                    severity = severity_scores[ci]
                    grounding_score = max(0.0, grounding_score - severity)
                    contradiction_details.append(contradictions[ci])
            contradiction_detail = "; ".join(contradiction_details)

            is_hallucination = grounding_score < 0.5

            detailed_findings.append(
                {
                    "finding": findings[i],
                    "target": anatomical_targets[i],
                    "status": clinical_statuses[i],
                    "baseline_classification": cls,
                    "baseline_reasoning": reason,
                    "pubmed_grounded": evidence == "supported",
                    "evidence_verdict": evidence,
                    "is_contradicted": is_contradicted,
                    "contradiction_detail": contradiction_detail,
                    "grounding_score": grounding_score,
                    "hallucination_risk": is_hallucination,
                }
            )

            # Update KG confidence
            finding_name = f"Claim_{i+1}"
            if finding_name in self.kg.entities:
                self.kg.entities[finding_name].confidence = grounding_score

            # Update edge confidence
            target_name = anatomical_targets[i].title()
            if target_name in self.kg.graph and finding_name in self.kg.graph[target_name]:
                self.kg.graph[target_name][finding_name]["confidence"] = grounding_score

        # Calculate overall verdict
        if detailed_findings:
            overall_confidence = sum(f["grounding_score"] for f in detailed_findings) / len(detailed_findings)
            hallucination_detected = any(f["hallucination_risk"] for f in detailed_findings)
        else:
            overall_confidence = 0.0
            hallucination_detected = False

        # Print per-finding verdict
        for i, item in enumerate(detailed_findings):
            verdict = "[FAIL] HALLUCINATED" if item["hallucination_risk"] else "[PASS] NOT HALLUCINATED"
            print(f"\n    Claim {i+1}: {verdict}  (Confidence: {item['grounding_score']:.2f})")
            print(f"      \"{item['finding']}\"")
            print(f"      Baseline: {item['baseline_classification']} | PubMed Grounded: {item['pubmed_grounded']} | Contradicted: {item['is_contradicted']}")
            if item["contradiction_detail"]:
                print(f"      Contradiction: {item['contradiction_detail']}")

        # A failed LLM step defaults its findings to "fine", so without a detected hallucination
        # the result is undetermined rather than a pass.
        if hallucination_detected:
            verdict = "hallucinated"
        elif llm_errors:
            verdict = "undetermined"
        else:
            verdict = "consistent"

        # Print overall verdict
        overall_verdict = {
            "hallucinated": "[FAIL] HALLUCINATION DETECTED",
            "undetermined": "[????] UNDETERMINED (incomplete check)",
            "consistent": "[PASS] REPORT IS CLINICALLY CONSISTENT",
        }[verdict]
        print(f"\n  {'-' * 50}")
        print(f"  OVERALL VERDICT: {overall_verdict}")
        print(f"  Average Confidence Score: {overall_confidence:.2f}")
        print(f"  Graph Stats: {len(self.kg.entities)} entities, {len(self.kg.relations)} relations")
        for err in llm_errors:
            print(f"  [WARN] NOT FULLY VERIFIED: {err}")
        print(f"  {'-' * 50}")

        return {
            "report": report,
            "overall_confidence": overall_confidence,
            "hallucination_detected": hallucination_detected,
            "verdict": verdict,
            "fully_verified": not llm_errors,
            "llm_errors": llm_errors,
            "detailed_findings": detailed_findings,
            "contradictions": contradictions,
            "graph_stats": {
                "num_entities": len(self.kg.entities),
                "num_relations": len(self.kg.relations),
            },
        }
