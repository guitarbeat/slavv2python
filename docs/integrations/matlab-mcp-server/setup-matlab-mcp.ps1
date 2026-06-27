<#
.SYNOPSIS
    Set up the official MATLAB MCP Server (MathWorks) for Claude Code.

.DESCRIPTION
    Validates the MATLAB root, ensures the server binary is present (downloading
    it from a release URL if one is supplied), then registers it with Claude Code
    via `claude mcp add`. Reusable across machines/projects.

    Requires MATLAB R2021a or later.

    PARITY WARNING (slavv2python): never use this server to regenerate the parity
    oracle — that must stay on hand-run R2019a .m scripts. Safe for ad-hoc MATLAB
    exploration only.

.PARAMETER MatlabRoot
    Path to the MATLAB install root, e.g. "C:\Program Files\MATLAB\R2026a".

.PARAMETER BinaryPath
    Where the server binary lives (or should be saved). Defaults to
    "C:\tools\matlab-mcp\matlab-mcp-server-windows-x64.exe".

.PARAMETER DownloadUrl
    Optional. If the binary is missing and this is provided, download it here.
    Find the current asset URL on:
    https://github.com/matlab/matlab-mcp-server/releases/latest

.PARAMETER ServerName
    Name to register the MCP server under (default: "matlab").

.EXAMPLE
    ./setup-matlab-mcp.ps1 -MatlabRoot "C:\Program Files\MATLAB\R2026a" -WhatIf

.EXAMPLE
    ./setup-matlab-mcp.ps1 `
      -MatlabRoot "C:\Program Files\MATLAB\R2026a" `
      -DownloadUrl "https://github.com/matlab/matlab-mcp-server/releases/latest/download/matlab-mcp-server-windows-x64.exe"
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$MatlabRoot,

    [string]$BinaryPath = "C:\tools\matlab-mcp\matlab-mcp-server-windows-x64.exe",

    [string]$DownloadUrl,

    [string]$ServerName = "matlab"
)

$ErrorActionPreference = "Stop"

# 1. Validate MATLAB root + version (must be R2021a or later)
if (-not (Test-Path $MatlabRoot)) {
    throw "MATLAB root not found: $MatlabRoot"
}
if ($MatlabRoot -match 'R20(\d{2})[ab]') {
    if ([int]$Matches[1] -lt 21) {
        throw "MATLAB $($Matches[0]) is below the R2021a minimum required by the MCP server."
    }
}
else {
    Write-Warning "Could not parse a release from '$MatlabRoot'; ensure it is R2021a or later."
}

# 2. Ensure the binary exists (download if a URL was given)
if (-not (Test-Path $BinaryPath)) {
    if ($DownloadUrl) {
        $dir = Split-Path -Parent $BinaryPath
        if (-not (Test-Path $dir)) {
            if ($PSCmdlet.ShouldProcess($dir, "Create directory")) {
                New-Item -ItemType Directory -Force -Path $dir | Out-Null
            }
        }
        if ($PSCmdlet.ShouldProcess($BinaryPath, "Download server binary")) {
            Write-Host "Downloading $DownloadUrl ..."
            Invoke-WebRequest -Uri $DownloadUrl -OutFile $BinaryPath
        }
    }
    else {
        throw "Binary not found at $BinaryPath and no -DownloadUrl given. " +
              "Download it from https://github.com/matlab/matlab-mcp-server/releases/latest"
    }
}

# 3. Confirm the claude CLI is available
$claude = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claude) {
    throw "The 'claude' CLI is not on PATH; cannot run 'claude mcp add'."
}

# 4. Register the server with Claude Code
$addArgs = @(
    "mcp", "add", "--transport", "stdio", $ServerName, "--",
    $BinaryPath, "--matlab-root=$MatlabRoot"
)
if ($PSCmdlet.ShouldProcess($ServerName, "claude mcp add")) {
    Write-Host "Registering MCP server '$ServerName' -> $BinaryPath (--matlab-root=$MatlabRoot)"
    & claude @addArgs
    Write-Host "Done. Verify with: claude mcp list"
}
else {
    Write-Host "[WhatIf] Would run: claude $($addArgs -join ' ')"
}
