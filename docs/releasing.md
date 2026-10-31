# Releasing

Releases are built and published by GitHub Actions when a tag `v*` is pushed. Publishing uses PyPI
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no token is stored anywhere.

## One-time setup (the repository owner)

1. On PyPI, create the project `airon`, or add a *pending publisher* if it does not exist yet:
   **Your account > Publishing > Add a new pending publisher**.
2. Fill in:
   - PyPI project name: `airon`
   - Owner: `Manifestro`
   - Repository name: `airon-metrics`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repository, **Settings > Environments > New environment** named `pypi`. Add
   *required reviewers* if every release should need an approval.

To rehearse on TestPyPI, repeat step 1 and 2 on test.pypi.org and temporarily point the workflow's
`repository-url` at `https://test.pypi.org/legacy/`.

## Making a release

1. Update `src/airon/_version.py` and `CHANGELOG.md` (move *unreleased* to the new version and date).
2. Commit and merge to `main`; wait for CI to pass.
3. Tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. The `Release` workflow builds the sdist and wheel, checks the tag matches the package version,
   runs `twine check`, and publishes (after approval, if you set required reviewers).
5. Verify in a clean environment: `pip install airon==0.1.0 && airon version`.

A mistake in a published version cannot be replaced: yank it on PyPI and release a new patch version.
