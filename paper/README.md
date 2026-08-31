# The JOSS submission

`paper.md` and `paper.bib` are a draft submission to the
[Journal of Open Source Software](https://joss.theoj.org/) for
`nerve_discover.h`.

## Before submitting

Two fields in the front matter are placeholders and must be filled in by hand:

| field | what it currently says | what it needs |
|-------|------------------------|---------------|
| `orcid` | `0000-0000-0000-0000` | a real ORCID iD — register free at <https://orcid.org> |
| `affiliations[0].name` | `Independent researcher` | whichever affiliation should appear |

Everything else — the performance table, the claims about what the engine
does — is generated from or checked against the repository, and is reproduced
by `make -C bench/feynman full`.

## Building it locally

JOSS compiles the paper itself, but the draft can be previewed with the
official container:

```sh
docker run --rm -v "$PWD/paper":/data -u "$(id -u):$(id -g)" \
  openjournals/inara -o pdf,crossref paper.md
```

## What JOSS will check

- The software is open source under an OSI-approved licence — Apache-2.0. ✔
- It has documentation, tests, and an issue tracker. ✔
- It represents a substantial scholarly contribution, not a thin wrapper.
  The argument for that is the Statement of Need: symbolic regression exists
  only as scientific-Python and Julia packages, and this is the same class of
  method in a form that can be embedded.
- The paper is short — JOSS wants a summary, not a full research article. The
  draft is deliberately near the upper end of their length guidance, so cut
  before adding.
