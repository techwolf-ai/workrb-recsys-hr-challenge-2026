# Skill extraction: approaches, vocabularies, and training data

A reference list of published work relevant to this challenge: the task
definitions the literature uses, the skill vocabularies, the available
training data, and the main method families with the defining paper for
each. It catalogs what exists; it does not rank approaches or recommend
one. The challenge [rules](../README.md#rules) allow anything that
produces rankings, whether or not it appears here.

Companion notebook: [`get_started.ipynb`](get_started.ipynb) for the
concrete task and metric setup as WorkRB measures it. The shipped
WorkRB baselines are instantiated in `participant/test.py`.

Umbrella survey: Senger et al., *Deep Learning-based Computational Job
Market Analysis* (NLP4HR 2024, [arXiv:2402.05617](https://arxiv.org/abs/2402.05617)).

## 1. Task definitions

Papers use these terms interchangeably; the differences matter when
reading results.

- **Skill extraction** (span detection). Find the contiguous tokens in
  a sentence that name a skill. Output is a BIO-tagged span. The
  classic formulation is SkillSpan (Zhang et al., NAACL 2022,
  [arXiv:2204.12811](https://arxiv.org/abs/2204.12811)).
- **Skill classification.** Given a sentence (or pre-extracted span),
  output one or more labels from a fixed taxonomy. When the label space
  is large enough that softmax over labels is impractical (~10K+
  labels), the task is **extreme multi-label classification (XMLC)**.
  ESCO (~13.9K skills) is one such label space; Bhola JD2Skills (20K)
  is another.
- **Skill entity linking / normalisation.** Map a free-text mention to
  its canonical taxonomy entry. The first sustained treatment is Zhang,
  van der Goot, Plank, *Entity Linking in the Job Market Domain* (EACL
  2024 Findings, [arXiv:2401.17979](https://arxiv.org/abs/2401.17979)).
- **Skill recommendation.** Given a job ad, a CV, or a course
  description, recommend which skills are required or implied. This is
  WorkRB's framing.

## 2. Vocabularies

- **[ESCO](https://esco.ec.europa.eu/en/classification)** (European
  Skills, Competences, Qualifications and Occupations). Two pillars:
  ~3K occupations and ~13.9K skills (v1.2.1, released late 2025), split
  into Knowledge, Skills, Language skills and knowledge, and
  Transversal skills. Multilingual (28 languages). Each concept has a
  preferred label, alternative labels, a one-paragraph description, and
  *broader / narrower / essential-for / optional-for* relations to
  occupations. Free and stable. **The WorkRB tasks target ESCO.**
- **[O\*NET](https://www.onetonline.org/)** is the US-government
  equivalent: ~1K occupations with per-occupation profiles of tasks,
  detailed work activities, and skill/knowledge/ability ratings.
  English only, finer granularity than ESCO. Cross-walks to ESCO exist
  (the ESCO data-science team publishes one).

## 3. Training data

| dataset | language | task | size | notes |
|---|---|---|---|---|
| [SkillSpan](https://huggingface.co/datasets/jjzha/skillspan) ([Zhang et al. 2022](https://arxiv.org/abs/2204.12811)) | en | span extraction | 14.5K sentences, 12.5K spans | hard and soft skills annotated separately |
| [Kompetencer](https://huggingface.co/datasets/jjzha/kompetencer) ([Zhang et al. 2022](https://arxiv.org/abs/2205.01381)) | da, en | span + ESCO link | distant supervision | English data is in the paper, HF release is Danish-only |
| [gnehm](https://huggingface.co/datasets/jjzha/gnehm) ([Gnehm et al. 2022](https://aclanthology.org/2022.nlpcss-1.2/)) | de | span (ICT) | ~25K sentences | German span-extraction set |
| [FIJO](https://github.com/iid-ulaval/FIJO-dataset) ([Beauchemin et al. 2022](https://arxiv.org/abs/2204.05208)) | fr | span (soft skills) | 867 ads | French resource |
| [Bhola JD2Skills](https://github.com/WING-NUS/JD2Skills-BERT-XMLC) (COLING 2020) | en | XMLC | 20K ads | 20K-label BERT-XMLC training set |
| [Synthetic-ESCO-skill-sentences](https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences) ([Decorte et al. 2023](https://arxiv.org/abs/2307.10778)) | en | retrieval / XMLC | 138K pairs | LLM-generated, covers ~all of ESCO v1.1.0; shipped with this challenge |
| [SkillSkape](https://github.com/magantoine/JobSkape) ([Magron et al. 2024](https://arxiv.org/abs/2402.03242)) | en | retrieval / XMLC | multi-skill sentences | synthetic, produced by the JobSkape pipeline |
| [Skill-XL](https://huggingface.co/datasets/TechWolf/Skill-XL) ([Decorte et al. 2025](https://arxiv.org/abs/2505.24640)) | en | sentence-level eval | 111 ads, 5.4K sentences, 8.5K skills | exhaustive sentence-level ESCO annotation; available as a WorkRB task |
| [Chinese-SkillSpan](https://arxiv.org/abs/2604.23009) (2026) | zh | span + ESCO link | 20K+ instances | Chinese recruitment platforms, ESCO-aligned |
| [UniSkill](https://arxiv.org/abs/2603.03134) (2026) | en | course-to-skill link | 2.2K pairs | university course learning goals linked to ESCO |
| [Tabiya/HaHu](https://arxiv.org/abs/2512.03195) (2025) | en, am | occupation linking | 542 ads + ~210K title pairs | Ethiopian job postings; job-title-to-ESCO-occupation pairs |

Two documented properties to be aware of when choosing data:

- Synthetic sets generated by prompting an LLM with the skill name tend
  to contain the label tokens verbatim, so token overlap becomes a
  strong predictor of a match. [`get_started.ipynb` §4](get_started.ipynb)
  demonstrates this on the shipped dataset; JobSkape
  ([arXiv:2402.03242](https://arxiv.org/abs/2402.03242)) analyses it
  and proposes an alternative generation pipeline.
- Span supervision and retrieval supervision carry different
  information: span-annotated sets have no ESCO labels (unless
  separately linked), and synthetic pair sets have no span boundaries.

On generating synthetic data: [arXiv:2601.09119](https://arxiv.org/abs/2601.09119)
(2026) generates contrastive pairs from ESCO definitions alone,
hierarchically constrained on ESCO Level-2 categories, and trains a
bi-encoder for Chinese job ads without labeled job-ad data.

## 4. Method families

Families in roughly chronological order. They are not mutually
exclusive; published systems often combine several.

### 4.1 Lexical

Sparse bag-of-words scoring: TF-IDF with cosine similarity, BM25 with
term saturation and length normalisation. No learning, no embeddings.
Shipped as the WorkRB baselines `TfIdfModel` and `BM25Model`.

### 4.2 Sequence labeling (span-based NER)

Token-level classification with BIO tags over a BERT-style encoder.

- **SkillSpan** (Zhang et al., NAACL 2022,
  [arXiv:2204.12811](https://arxiv.org/abs/2204.12811)): the dataset
  plus BERT / SpanBERT / JobBERTa baselines.
- **ESCOXLM-R** (Zhang, van der Goot, Plank, ACL 2023,
  [arXiv:2305.12092](https://arxiv.org/abs/2305.12092)): XLM-R
  continued-pretrained on ESCO concept relations. Checkpoints:
  [`jjzha/esco-xlm-roberta-large`](https://huggingface.co/jjzha/esco-xlm-roberta-large),
  `jjzha/escoxlmr_skill_extraction`, `jjzha/escoxlmr_knowledge_extraction`.
- **NNOSE** (Zhang, van der Goot, Kan, Plank, EACL 2024,
  [arXiv:2401.17092](https://arxiv.org/abs/2401.17092)): kNN-augmented
  sequence labeler over a datastore combining multiple span datasets.

### 4.3 Bi-encoder + contrastive learning

Encode sentence and skill into the same vector space; score by cosine
similarity; train with a contrastive loss.

- **Sentence-BERT** (Reimers and Gurevych, EMNLP 2019,
  [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)): the base
  recipe.
- Decorte et al., *Design of Negative Sampling Strategies for Distantly
  Supervised Skill Extraction* (RecSys-in-HR 2022,
  [arXiv:2209.05987](https://arxiv.org/abs/2209.05987)): hard negatives
  drawn from ESCO siblings.
- Decorte et al., *Extreme Multi-Label Skill Extraction Training using
  Large Language Models* (AI4HR&PES @ ECML-PKDD 2023,
  [arXiv:2307.10778](https://arxiv.org/abs/2307.10778)): InfoNCE with
  LLM-generated synthetic positives and ESCO-related negatives; the
  recipe behind the shipped synthetic dataset.
- **UWE**, *Unified Work Embeddings* (TechWolf,
  [arXiv:2511.07969](https://arxiv.org/abs/2511.07969)): a 109M MPNet
  bi-encoder trained multi-task across work-domain ranking tasks
  (skills, jobs, normalisation, similarity).

### 4.4 Token-level scoring / late interaction

Keep per-token vectors instead of mean-pooling; at scoring time the
skill representation attends over sentence tokens.

- **ColBERT** (Khattab and Zaharia, SIGIR 2020,
  [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)): the canonical
  IR reference for late interaction.
- **ConTeXT-Match** (Decorte et al., *Efficient Text Encoders for Labor
  Market Analysis*, IEEE Access 2025,
  [arXiv:2505.24640](https://arxiv.org/abs/2505.24640)): late
  interaction adapted to sentence-to-ESCO retrieval. Checkpoint:
  [`TechWolf/ConTeXT-Skill-Extraction-base`](https://huggingface.co/TechWolf/ConTeXT-Skill-Extraction-base);
  shipped as the WorkRB baseline `ConTeXTMatchModel`.

### 4.5 Two-stage retrieve then rerank

A cheap retriever shortlists top-K candidates from the full vocabulary;
a cross-encoder rescores each (sentence, skill) pair jointly.

- Bielinski and Brazier, *From Retrieval to Ranking: A Two-Stage Neural
  Framework for Automated Skill Extraction* (RecSys-in-HR 2025,
  [CEUR Vol-4046](https://ceur-ws.org/Vol-4046/RecSysHR2025-paper_5.pdf)):
  curriculum-trained bi-encoder retrieval plus cross-encoder ranking.
  Released stage-1 retriever:
  [`Aleksandruz/skillmatch-mpnet-curriculum-retriever`](https://huggingface.co/Aleksandruz/skillmatch-mpnet-curriculum-retriever)
  (the stage-2 reranker is not released); shipped as the WorkRB
  baseline `CurriculumMatchModel`.

### 4.6 Decoder-LLM and generative approaches

- **IReRa** (D'Oosterlinck et al., *In-Context Learning for Extreme
  Multi-Label Classification*, NeurIPS 2024,
  [arXiv:2401.12178](https://arxiv.org/abs/2401.12178)): three frozen
  modules wired in DSPy (LLM infer, dense retrieve, LLM rerank), ~50
  labeled examples for prompt optimisation. Code:
  [`KarelDO/xmc.dspy`](https://github.com/KarelDO/xmc.dspy).
- Clavié and Soulié, *Large Language Models as Batteries-Included
  Zero-Shot ESCO Skills Matchers* (RecSys-in-HR 2023,
  [arXiv:2307.03539](https://arxiv.org/abs/2307.03539)): classifier +
  similarity retriever + GPT-4 rerank.
- **SkiLLens** ([EACL 2026 Industry Track](https://aclanthology.org/2026.eacl-industry.65/)):
  multilingual industrial system; mines candidate skill terms per
  country, links to ESCO with `gte-large` cosine retrieval, GPT-4
  selects among the top 3.
- **Skill-LLM** (Herandi et al.,
  [arXiv:2410.12052](https://arxiv.org/abs/2410.12052)): LoRA-fine-tuned
  Llama-3-8B emitting BIO-tagged spans plus ESCO label.
- **SRICL** (Li et al., [arXiv:2604.21525](https://arxiv.org/abs/2604.21525)):
  semantic retrieval + ICL + SFT with a deterministic verifier
  enforcing BIO legality, non-overlap, and pairing.
- Zhang, van der Goot, Plank, *Entity Linking in the Job Market Domain*
  (EACL 2024 Findings, [arXiv:2401.17979](https://arxiv.org/abs/2401.17979)):
  encoder (BLINK) vs autoregressive (GENRE) linking of pre-extracted
  spans to ESCO.
- Tabiya/HaHu ([arXiv:2512.03195](https://arxiv.org/abs/2512.03195)):
  compares direct sentence linking against extract-then-link on
  Ethiopian job postings, for both skills and occupations.
- Nguyen, Zhang, Montariol, Bosselut, *Rethinking Skill Extraction in
  the Job Market Domain using LLMs* (NLP4HR 2024,
  [arXiv:2402.03832](https://arxiv.org/abs/2402.03832)): few-shot ICL
  versus supervised span extractors across six uniformised datasets.
- Dukić and Šnajder, *Do Not (Always) Look Right* (EACL 2024,
  [arXiv:2401.14556](https://arxiv.org/abs/2401.14556)): layer-wise
  causal-mask removal to use decoder LLMs as sequence labelers.

### 4.7 Graph- and hierarchy-aware

Methods that use the ESCO graph (broader/narrower skills,
essential-for/optional-for relations to occupations) or a comparable
taxonomy structure.

- **ESCOXLM-R** (ACL 2023, [arXiv:2305.12092](https://arxiv.org/abs/2305.12092)):
  taxonomical-relation pretraining loss, see 4.2.
- **Labor Space** (Kim, Ahn, Park, 2023,
  [arXiv:2311.06310](https://arxiv.org/abs/2311.06310)): a unified
  embedding space for occupations, skills, industries, and firms;
  heterogeneous-graph framing.
- **Mind the Task Gap** (Tan et al., RecSys-in-HR 2025,
  [CEUR Vol-4046](https://ceur-ws.org/Vol-4046/)): unsupervised
  skill-task link prediction with a variational graph auto-encoder over
  Singapore's Skills Framework (1.9K roles, 27K tasks, 3.1K skills).
