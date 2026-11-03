# ParlIR data licence

The ParlIR code is MIT-licensed (`LICENSE`). This file covers the **data**.

## Source material

| Material | Rights holder | Terms |
|---|---|---|
| Written questions, metadata, Q↔A links (EP Open Data Portal) | European Parliament | CC BY 4.0 — Decision of the Bureau of 16 December 2024 on the European Parliament's Open Data (OJ C/2025/341), Art. 4 |
| Answers given by the European Commission | European Union / Commission | Commission Decision 2011/833/EU on the reuse of Commission documents; CC BY 4.0 adopted as the default open licence by Decision C(2019) 1655 |
| Answers given by the Council | Council of the EU | Council Decision (EU) 2017/1842 on the open data policy of the Council and the reuse of Council documents |

Reuse conditions common to these instruments: acknowledge the source, do not distort the original meaning, and
the institutions accept no liability for the consequences of reuse. ParlIR **modifies** the texts (header
stripping, identifier masking, text extraction from XHTML/DOCX/XML); `raw_text` keeps the header-stripped,
unmasked body.

## ParlIR annotations

Splits, masking, extracted citation edges (`precedent_meta.jsonl`), derived qrels and build manifests are
released under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).

## Attribution statement

Please include the following when redistributing ParlIR data or derivatives:

> Contains European Parliament written questions and official answers. Source: European Parliament Open Data
> Portal (data.europarl.europa.eu), © European Union, 2014–2026, CC BY 4.0. Commission answers reused under
> Decision 2011/833/EU (CC BY 4.0); Council answers reused under Decision (EU) 2017/1842. Texts were modified
> (header removal, identifier masking). ParlIR annotations © 2026 Marek Šuppa and Viktória Ondrejová, CC BY 4.0.

## Personal data

Author names are those of Members of the European Parliament acting in their official capacity, as published by
the Parliament. They are kept as metadata and removed from the canonical retrieval text where they appear in
headers and co-signatory lines; residual mentions remain (e.g. 12 train queries with a "Question supported by …"
list, and MEPs or other public figures named in question/answer content). `subject`, `header` and `raw_text` are
unmasked. See docs/datasheet.md "Known issues in v1.0". Removal requests concerning personal data can be filed on
the code repository's issue tracker (https://github.com/NaiveNeuron/ParlIR-ecir-2027/issues). The maintainers review
each request. Accepted removals apply to the main branch and the next release, while published, pinned revisions
stay unchanged so that reported results remain reproducible, unless the law requires otherwise.
