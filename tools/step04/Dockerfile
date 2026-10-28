# Subset of Miles 5078ff7's recipe needed by the FSDP and serving gates.
ARG BASE=lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
FROM ${BASE}
ENV UV_NO_SYNC=1 SGLANG_BUILD_RUST_EXTS=none
WORKDIR /workspace
# Source installs happen at launch from persistent, revision-checked mounts.
