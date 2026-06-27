# MATLAB MCP Server — Setup Guide

Reusable setup notes for wiring the **official MathWorks MATLAB MCP Server**
into Claude Code (or any MCP-capable agent). Kept git-tracked so it is ready
for a future MATLAB version, a new machine, or another project.

> **Status in this repo: NOT installed (intentionally).** See the parity
> warning below — the server cannot safely generate this project's parity
> oracle. These notes exist for future/other use only.

- Official server: <https://github.com/matlab/matlab-mcp-server>
- Product page: <https://www.mathworks.com/products/matlab-mcp-server.html>
- Background: [MathWorks blog — Claude Code + MATLAB MCP](https://blogs.mathworks.com/simulink/2026/02/26/simulink-and-simscape-modeling-using-claude-code-and-the-matlab-mcp-server/)

---

## ⚠️ Parity warning (read before using in slavv2python)

This project's MATLAB→Python certification is built on the **R2019a** oracle.
MATLAB floating-point results can differ subtly between releases, so generating
the oracle through **any** other MATLAB version risks silently breaking parity.

- ✅ Safe uses: exploring / running the MATLAB reference code ad-hoc
  (e.g. `external/Vectorization-Public`), prototyping, learning.
- ❌ Never use the MCP server to regenerate the parity oracle. The oracle must
  stay on hand-run **R2019a** `.m` scripts (see `workspace/scratch/*.m`).

---

## Requirements

| Requirement | Detail |
|---|---|
| MATLAB version | **R2021a or later** (server supports ~5 most recent releases) |
| MATLAB on PATH | `matlab` must resolve, or pass `--matlab-root` explicitly |
| Claude Code CLI | `claude` on PATH (for `claude mcp add`) |
| OS binary | `matlab-mcp-server-windows-x64.exe` (Windows x64) from the releases page |

> On this machine the installed MATLAB releases are R2017b / R2019a / R2020b —
> **all below R2021a**, so the official server will not run here today. Revisit
> after installing R2021a+.

---

## Quick setup (Windows + Claude Code)

1. **Download the binary** for your OS from the latest release:
   <https://github.com/matlab/matlab-mcp-server/releases/latest>
   (asset: `matlab-mcp-server-windows-x64.exe`). Save it somewhere stable, e.g.
   `C:\tools\matlab-mcp\matlab-mcp-server-windows-x64.exe`.

2. **Register it with Claude Code**, pointing `--matlab-root` at your MATLAB
   install (adjust the version folder):

   ```powershell
   claude mcp add --transport stdio matlab -- `
     "C:\tools\matlab-mcp\matlab-mcp-server-windows-x64.exe" `
     --matlab-root="C:\Program Files\MATLAB\R2026a"
   ```

3. **Verify**:

   ```powershell
   claude mcp list
   ```

   The `matlab` server should appear and connect.

### Per-project (committed) config alternative

Instead of `claude mcp add` (which writes to your user config), you can commit a
project-scoped server by copying [`mcp.example.json`](./mcp.example.json) to a
`.mcp.json` at the repo root and editing the two paths. A project `.mcp.json` is
picked up automatically by Claude Code for that repo.

> In slavv2python, do **not** commit a live `.mcp.json` for this server while the
> parity work is active — keep it disabled per the warning above.

---

## Helper script

[`setup-matlab-mcp.ps1`](./setup-matlab-mcp.ps1) automates steps 1–3: it
validates the MATLAB root, ensures the binary exists (downloading it if a URL is
provided), and runs `claude mcp add`. Run with `-WhatIf` first to preview.

```powershell
./setup-matlab-mcp.ps1 -MatlabRoot "C:\Program Files\MATLAB\R2026a" -WhatIf
```

---

## Removal

```powershell
claude mcp remove matlab
```

(Or delete the `matlab` entry from `.mcp.json` if you used the project-scoped
config.)
