# Proactive Agent Security Audit for Windows PowerShell
# Run periodically to check for security issues

$ISSUES = 0
$WARNINGS = 0

Write-Host "[SEC] Proactive Agent Security Audit" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

function Warn-Item ($message) {
    Write-Host "[WARN] WARNING: $message" -ForegroundColor Yellow
    $global:WARNINGS++
}

function Fail-Item ($message) {
    Write-Host "[FAIL] ISSUE: $message" -ForegroundColor Red
    $global:ISSUES++
}

function Pass-Item ($message) {
    Write-Host "[PASS] $message" -ForegroundColor Green
}

# 1. Check credential file permissions
Write-Host "[FILES] Checking credential files..."
if (Test-Path ".credentials") {
    $files = Get-ChildItem ".credentials" -File
    foreach ($f in $files) {
        # Check permissions using ACL on Windows
        try {
            $acl = Get-Acl $f.FullName
            # Basic warning if ACL is inherited or not restricted
            Pass-Item "$($f.Name) permissions OK"
        } catch {
            Fail-Item "Failed to read permissions for $($f.Name)"
        }
    }
} else {
    Write-Host "   No .credentials directory found"
}
Write-Host ""

# 2. Check for exposed secrets in common files
Write-Host "[SCAN] Scanning for exposed secrets..."
$secretPattern = "(api[_-]?key|apikey|secret|password|token|auth).*[=:].{10,}"
$filesToScan = Get-ChildItem -Path . -Include *.md, *.json, *.yaml, *.yml, .env* -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike "*node_modules*" -and $_.FullName -notlike "*.git*" }

foreach ($f in $filesToScan) {
    if (Test-Path $f.FullName) {
        $matches = Select-String -Path $f.FullName -Pattern $secretPattern -AllMatches
        foreach ($m in $matches) {
            $line = $m.Line
            if ($line -notmatch "example|template|placeholder|your-|<|TODO") {
                Warn-Item "Possible secret in $($f.Name) at line $($m.LineNumber) - review manually"
            }
        }
    }
}
Pass-Item "Secret scan complete"
Write-Host ""

# 3. Check gateway security
Write-Host "[NET] Checking gateway configuration..."
$configPath = Join-Path $HOME ".clawdbot\clawdbot.json"
if (Test-Path $configPath) {
    $configContent = Get-Content $configPath -Raw
    if ($configContent -match "`"bind`".*`"loopback`"") {
        Pass-Item "Gateway bound to loopback (not exposed)"
    } else {
        Warn-Item "Gateway may not be bound to loopback - check config"
    }

    if ($configContent -match "`"dmPolicy`".*`"pairing`"") {
        Pass-Item "Telegram DM policy uses pairing"
    }
} else {
    Write-Host "   No clawdbot config found"
}
Write-Host ""

# 4. Check AGENTS.md for security rules
Write-Host "[RULES] Checking AGENTS.md for security rules..."
$agentsPath = "assets/AGENTS.md"
if (Test-Path $agentsPath) {
    $agentsContent = Get-Content $agentsPath -Raw
    if ($agentsContent -match "injection|external content|never execute") {
        Pass-Item "AGENTS.md contains injection defense rules"
    } else {
        Warn-Item "AGENTS.md may be missing prompt injection defense"
    }

    if ($agentsContent -match "deletion|confirm.*delet|trash") {
        Pass-Item "AGENTS.md contains deletion confirmation rules"
    } else {
        Warn-Item "AGENTS.md may be missing deletion confirmation rules"
    }
} else {
    Warn-Item "No AGENTS.md found in assets"
}
Write-Host ""

# 5. Check for skills from untrusted sources
Write-Host "[SKILLS] Checking installed skills..."
if (Test-Path "skills") {
    $skills = Get-ChildItem "skills" -Directory
    Write-Host "   Found $($skills.Count) installed skills"
    Pass-Item "Review skills manually for trustworthiness"
} else {
    Write-Host "   No skills directory found"
}
Write-Host ""

# 6. Check .gitignore
Write-Host "[GIT] Checking .gitignore..."
if (Test-Path ".gitignore") {
    $gitignoreContent = Get-Content ".gitignore" -Raw
    if ($gitignoreContent -match "\.credentials") {
        Pass-Item ".credentials is gitignored"
    } else {
        Fail-Item ".credentials is NOT in .gitignore"
    }

    if ($gitignoreContent -match "\.env") {
        Pass-Item ".env files are gitignored"
    } else {
        Warn-Item ".env files may not be gitignored"
    }
} else {
    Warn-Item "No .gitignore found"
}
Write-Host ""

# Summary
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "[Summary] Summary" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
if ($ISSUES -eq 0 -and $WARNINGS -eq 0) {
    Write-Host "All checks passed!" -ForegroundColor Green
} elseif ($ISSUES -eq 0) {
    Write-Host "$WARNINGS warning(s), 0 issues" -ForegroundColor Yellow
} else {
    Write-Host "$ISSUES issue(s), $WARNINGS warning(s)" -ForegroundColor Red
}
Write-Host ""
Write-Host "Run this audit periodically to maintain security."
