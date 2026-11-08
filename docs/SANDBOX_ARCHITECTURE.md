# CDM-1 Execution Sandbox Architecture & Deployment Guide

This document specifies the security isolation model for CDM-1 tool execution, container sandboxing, and production deployment topology.

---

## 1. Security Isolation Principles

The `shell_exec` execution tool provides code and command execution for agent loops with strict defense-in-depth:

1. **Zero Host Execution**:
   - The shell tool will **never** fall back to executing commands on the host operating system.
   - If Docker or the container runtime is unavailable, execution fails with a structured error and refuses to run.

2. **Pinned Image Digest**:
   - The sandbox execution image is pinned to an immutable digest (`alpine@sha256:beefdbd8a1da6d2915566fde36db9db0b524eb737fc57cd13675d29e083a2072`), preventing image tampering and upstream tag mutation.

3. **Restricted Container Capabilities**:
   - `--network none`: Complete network isolation. Outbound network requests, command-and-control beacons, and exfiltration attempts fail unconditionally.
   - `--read-only`: The container root filesystem is mounted read-only.
   - `--cap-drop ALL`: All Linux capabilities are dropped.
   - `--security-opt no-new-privileges`: Prevents privilege escalation via `setuid` or `setgid` binaries.
   - `--tmpfs /tmp:rw,noexec,nosuid,size=32m`: Ephemeral scratch storage with execution disabled.
   - `--user 1000:1000`: Commands run strictly as an unprivileged non-root user.
   - Resource limits: memory (`--memory`), CPU quotas (`--cpus`), and process limits (`--pids-limit`) prevent fork bombs and denial of service.

---

## 2. Supported Deployment Topologies

### Why `/var/run/docker.sock` is NOT Mounted into the API Container
Mounting the host Docker socket (`/var/run/docker.sock`) into an application web container grants equivalent root privileges on the host system. If the API were compromised, an attacker could trivially escape into the host by spawning a privileged container.

Therefore:
- In the default Docker Compose deployment, `/var/run/docker.sock` is **not** mounted.
- When the API container runs without host Docker access, `shell_exec` reports `unavailable` with a structured error.

### Supported Production Topologies
1. **Host-Level Deployment (Direct Agent Execution)**:
   - Run the CDM-1 service directly on a secured Linux virtual machine where the service user has access to Docker. The service will spawn isolated sibling containers using the Docker CLI.
2. **Dedicated Sandbox Runner Service**:
   - The CDM-1 API communicates over gRPC/mTLS with a dedicated, isolated sandbox runner service located in an isolated subnet.
   - The runner service manages ephemeral microVMs (e.g. AWS Firecracker, Kata Containers) or ephemeral Kubernetes sandbox pods (`gVisor`/`runsc`) with per-tenant lifecycle tracking.
