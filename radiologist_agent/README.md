# Radiologist Reporting Agent

An agentic workflow that assembles a radiology report from the clinical source
systems, sharpens it with medical evidence and current guidelines, and — when a
critical finding is present — runs closed-loop communication with the care team
until the result is acknowledged.

Where Tricorder **triages** studies into the reading queue, this agent picks up
**after** the read: it drafts the report and closes the loop on critical-result
communication. It is a self-contained package and does not modify any existing
Tricorder code.

> ⚠️ **RESEARCH PROTOTYPE — NON-DIAGNOSTIC.** Drafts and routes; a radiologist
> owns the final report and the communication.

## What it does

The pipeline (`orchestrator.py`) runs these steps in order:

1. **Bridge** — pulls the clinical **history** from the order/HL7 note.
2. **Tech sheet** — pulls the **technique** and **contrast dose**.
3. **PACS** — pulls the prior-study **comparison**.
4. **Radiology model** — suggests a **template**, **findings**, and **impression**.
5. **Dictation** — prompts the radiologist to add their **own findings**.
6. **Report tree** — presents a **tree of options** for building the final report
   (findings source, impression source, recommendations).
7–9. Assembles findings/impression, **integrates medical evidence** into the
   impression, and derives **recommendations from current guidelines**.
10. **Critical detection** — flags any **critical communicable findings**.
11. **Care team** — if critical, finds the **ordering physician**, the
    **primary care provider** (from recent notes), and the **floor nurse**
    (from the chart).
12. **Notify** — sends secure messages to **all three**.
13. **Escalate / confirm** — if messages go **unacknowledged**, calls the floor
    by **phone** or **alerts the radiologist**; if communication **succeeds**, it
    says so.

## Layout

```
radiologist_agent/
  orchestrator.py        end-to-end pipeline (the 13 steps)
  cli.py / __main__.py   entry point
  models.py              typed data structures
  llm.py                 Claude wrapper (claude-opus-4-8) with offline fallback
  report.py              the report-construction decision tree
  critical.py            critical-finding classification
  data_sources/          mock integrations to real hospital systems
    bridge · tech_sheet · pacs · rad_model · chart
  knowledge/
    evidence.py          evidence-informed impression   (Claude)
    guidelines.py        guideline-based recommendations (Claude)
  communication/
    messaging.py         secure messaging + acknowledgement
    phone.py             phone escalation + radiologist alert
  data/case_ctpa.json    sample case (CTPA showing acute PE)
```

Each `data_sources/*` module stands in for a real HL7 / DICOM / FHIR
integration; swapping in the live endpoint leaves the orchestrator unchanged.

## Reasoning backend

The open-ended judgement steps (evidence synthesis, guideline recommendations,
critical-finding detection) call **Claude (`claude-opus-4-8`)**. If no
credentials are configured the tool falls back to a deterministic offline mode
so the full pipeline still runs — the console prints which backend is active.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # optional; runs offline without it
```

## Run

```bash
python -m radiologist_agent                 # bundled CTPA demo case
python -m radiologist_agent --interactive   # navigate the report tree + dictate
python -m radiologist_agent --case path.json
```

The demo case is a CT pulmonary angiogram showing an **acute pulmonary
embolism** — a critical finding — so it exercises the full detect → notify →
escalate → confirm communication path.
