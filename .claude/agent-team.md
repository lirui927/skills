---
repo: lirui927/skills
integration_branch: main
never_touch: jgong5/skills
gate_task: python3 -m pytest tests -q --ignore-glob='tests/*/test_make_pdf.py'
design_entry: CLAUDE.md
new_tests_dir: tests
prose_tests: tests/*/test_skill_md.py, tests/*/test_reference_docs.py, tests/asm_tutorial/test_cdna_facts.py
---
`gate_task` leaves out `tests/*/test_make_pdf.py`: those tests need pandoc,
a Chrome-family binary and poppler, which the dev container lacks. A change
to a skill's `make_pdf.py`, `check_pdf.py` or `tutorial.css` runs them where
those tools exist, and its PR states the result and the commit.

A change to `.claude-plugin/marketplace.json` also runs `claude plugin
validate .` and `claude plugin details <bundle>` (CLAUDE.md, Commands).

Run commands the way `CLAUDE.local.md` says when it exists.
