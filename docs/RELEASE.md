# Release checklist

## Required evidence

- CPU unit tests and C++ host tests pass in GitHub Actions.
- `make demo` succeeds from a clean checkout.
- Training/evaluation reports identify model, manifest, and artifact checksums.
- ONNX numeric parity and text/CER reports are attached.
- ATC log and model manifest identify CANN/ATC, SoC, input shape, and OM checksum.
- Atlas smoke report has `verified_on_device: true`.
- No raw audio, model binaries, tokens with restrictive licenses, secrets, or
  machine-specific paths are committed.

## Versioning

Tag source code as `vMAJOR.MINOR.PATCH`. Store large model artifacts in a
versioned release or approved artifact store, not Git history. Each model
artifact must be accompanied by the generated manifest and its training/export
provenance.

## Pull Request policy

Merge feature branches through a reviewed Pull Request. CPU CI is mandatory.
The manual Atlas workflow is mandatory when a change touches the OM contract,
frontend, decoder, AscendCL lifecycle, CMake, CANN scripts, or board runner.
Squash or rebase noisy work-in-progress commits before merge, and never merge a
release with unverified board performance claims.
