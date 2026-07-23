# Shared Atlas development board

One Atlas 200I DK A2 is enough for the project if it is treated as a shared test node rather than a shared source-code editor.

## Responsibility split

Work that runs without the board:

- dataset tools and tests;
- model training and CER evaluation;
- ONNX export and CPU inference;
- frontend and decoder unit tests;
- web UI and reporting;
- Pull Request review.

Work that requires the board:

- CANN and AscendCL compatibility;
- OM loading and execution;
- ONNX/OM output comparison;
- NPU latency, memory, temperature, and power measurements;
- long-running stability tests.

## Recommended access

1. Put the board and trusted collaborators on a private overlay network such as Tailscale.
2. Do not expose port 22 directly to the public internet.
3. Give every collaborator a separate non-root account or SSH identity.
4. Reserve driver changes, reboot, and device administration for the board owner.
5. Use Git branches and Pull Requests; do not edit a shared checkout.

VS Code Remote SSH can be used for interactive diagnosis. Routine tests should run through a GitHub Actions self-hosted runner.

## Runner registration

Register the board at repository or organization level and assign these labels:

```text
self-hosted
linux
arm64
atlas-310b
```

The repository includes `.github/workflows/atlas-smoke.yml`. It is manual-only so unreviewed public Pull Requests cannot automatically execute code on the board.

Before enabling the workflow, create stable local resources on the board:

```text
/opt/incar-asr/models/paraformer.om
/opt/incar-asr/models/tokens.txt
/opt/incar-asr/samples/smoke.wav
```

The runner account needs read access to those files and access to the Ascend device. It should not receive unrestricted `sudo`.

## Test policy

- Pull Requests run CPU tests first.
- A trusted collaborator manually starts the Atlas smoke workflow.
- Only one Atlas job runs at a time.
- The workflow checks out an exact Git ref.
- Logs are uploaded as workflow artifacts.
- Every run writes checksums, exit codes, detected NPU information, parsed
  latency/RTF, and `verified_on_device` to `report.json`.
- A failed Atlas result blocks release, not ordinary CPU development.

## Security warning

A self-hosted runner executes repository code on the board. For a public repository:

- never trigger it automatically for forked Pull Requests;
- restrict workflow modification through review;
- store credentials in GitHub Secrets, not shell scripts;
- avoid keeping unrelated personal data or reusable credentials on the runner;
- rotate runner credentials if untrusted code is executed.
