[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$inputText = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($inputText)) {
  exit 0
}

try {
  $payload = $inputText | ConvertFrom-Json -ErrorAction Stop
} catch {
  exit 0
}

# Best-effort extraction of the tool name and command text.
$toolName = $payload.toolName
$commandText = $null

if ($payload.toolInput -and $payload.toolInput.command) {
  $commandText = [string]$payload.toolInput.command
} elseif ($payload.toolArguments -and $payload.toolArguments.command) {
  $commandText = [string]$payload.toolArguments.command
}

if ($toolName -ne 'run_in_terminal' -or [string]::IsNullOrWhiteSpace($commandText)) {
  exit 0
}

$normalized = ($commandText -replace "\s+", " ").Trim().ToLowerInvariant()

# Heuristics: warn on common destructive operations.
$patterns = @(
  '(^|\s)rm(\s|$)',
  '(^|\s)rmdir(\s|$)',
  '(^|\s)del(\s|$)',
  '(^|\s)erase(\s|$)',
  'remove-item\b',
  'git\s+reset\s+--hard\b',
  'git\s+clean\b.*\s-f',
  'format-volume\b',
  'diskpart\b'
)

$matched = $false
foreach ($pattern in $patterns) {
  if ($normalized -match $pattern) {
    $matched = $true
    break
  }
}

if (-not $matched) {
  exit 0
}

$warningMessage = @(
  'Potentially destructive terminal command detected.',
  "Tool: $toolName",
  "Command: $commandText",
  'If this is intentional, proceed carefully (prefer non-destructive alternatives).'
) -join "`n"

$out = @{
  systemMessage = $warningMessage
}

$out | ConvertTo-Json -Depth 6
exit 0
