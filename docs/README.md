# Documentation

This directory contains the formal write-up for the agentic multimodal Foley system.

The main report source is at `docs/report/agentic_multimodal_report.tex`. The report includes a centered title page, table of contents, architecture and control-loop diagrams, detailed narrative sections on model selection and agent behavior, and a references section.

To build the PDF from repository root, run:

```bash
cd docs/report
pdflatex agentic_multimodal_report.tex
pdflatex agentic_multimodal_report.tex
```

If `latexmk` is installed, a one-command build is:

```bash
cd docs/report
latexmk -pdf agentic_multimodal_report.tex
```
