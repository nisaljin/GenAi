# Documentation

This directory contains the formal write-up for the agentic multimodal Foley system.

The main report source is at `docs/report/agentic_multimodal_report.tex`. The report includes a centered title page, table of contents, architecture and control-loop diagrams, detailed narrative sections on model selection and agent behavior, and a references section.

The Beamer slide deck source is at `docs/agentic_multimodal_foley_beamer.tex`. It is a standalone presentation version of the same system story, organized for a 4-speaker technical talk.

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

To build the slide deck:

```bash
cd docs
pdflatex agentic_multimodal_foley_beamer.tex
pdflatex agentic_multimodal_foley_beamer.tex
```

If `latexmk` is installed:

```bash
cd docs
latexmk -pdf agentic_multimodal_foley_beamer.tex
```
