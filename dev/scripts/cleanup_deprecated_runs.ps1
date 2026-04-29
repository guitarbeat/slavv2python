# Cleanup script for deprecated parity experiment runs
# Generated based on v22-pointer-corruption-fix spec review
# Total estimated space to free: ~3.5 GB

$baseDir = "D:\slavv_comparisons\experiments\live-parity\runs"

Write-Host "=== Cleanup Plan ===" -ForegroundColor Cyan
Write-Host "This will delete deprecated runs and free approximately 3.5 GB"
Write-Host ""

# CRITICAL: Keep these runs
$keepRuns = @(
    "20260421_accepted_budget_trial",  # Referenced in spec tasks 4.1-4.5
    "20260423_exact_proof_trial_v5",   # Latest exact proof trial
    "20260424_exact_boundary_trial",   # Recent boundary work
    "20260427_v22_fixed",              # Fixed code run
    "20260427_v22_pointer_fix_verification"  # Verification run
)

Write-Host "KEEPING (referenced in active work):" -ForegroundColor Green
$keepRuns | ForEach-Object { Write-Host "  - $_" }
Write-Host ""

# Category 1: Debug/test runs from v22 work (tiny files, <1 MB total)
$debugRuns = @(
    "20260427_v22_debug",
    "20260427_v22_debug2",
    "20260427_v22_trace_test",
    "20260427_v22_readback_test",
    "20260427_v22_overwrite_test",
    "20260427_v22_audit",
    "20260427_v22_audit_v2",
    "20260427_v22_pointer_fix_test"
)

# Category 2: Superseded candidate check runs (many small duplicates)
$candidateCheckRuns = @(
    "candidate_check_20260425",
    "candidate_check_20260425_v2",
    "candidate_check_20260425_v3",
    "candidate_check_20260425_v4",
    "candidate_check_20260426_v5",
    "candidate_check_20260426_v6",
    "candidate_check_20260426_v7",
    "candidate_check_20260426_v8",
    "candidate_check_20260426_v9",
    "candidate_check_20260426_v10",
    "candidate_check_20260426_v11",
    "candidate_check_20260426_v12",
    "candidate_check_20260426_v13",
    "candidate_check_20260426_v14",
    "candidate_check_20260426_v15",
    "candidate_check_20260426_v16",
    "candidate_check_20260426_v17",
    "candidate_check_20260426_v18",
    "candidate_check_20260426_v19",
    "final_check_20260426_v20",
    "final_check_20260426_v21",
    "final_check_20260426_v22",
    "20260423_capture_candidates_trial"
)

# Category 3: Old agent test runs
$agentTestRuns = @(
    "agent_native_parity_20260424_195721",
    "agent_native_parity_20260424_200006",
    "agent_native_parity_20260424_200315"
)

# Category 4: Superseded exact proof trials (keeping only v5)
$oldExactProofRuns = @(
    "20260422_exact_proof_trial",
    "20260423_exact_proof_trial_v2",
    "20260423_exact_proof_trial_v3",
    "20260423_exact_proof_trial_v4"
)

# Category 5: Old 20260421 trials (superseded by accepted_budget_trial)
$old20260421Runs = @(
    "20260421_all_contacts_trial",
    "20260421_origin_cap_trial",
    "20260421_child_override_trial",
    "20260421_frontier_restore_trial",
    "20260421_current_code_edges_rerun",
    "20260421_current_code_edges_rerun_v2",
    "20260421_current_code_edges_rerun_v3",
    "20260421_live_parity_clean"
)

# Category 6: Old cleanup and scratch runs
$oldMiscRuns = @(
    "20260422_exact_cleanup_trial",
    "20260420_live_parity_scratch",
    "20260420_live_parity_postmove",
    "20260418_claim_ordering_trial",
    "20260418_network_gate_trial",
    "20260401_live_parity_retry",
    "20260423_fail_fast_smoke"
)

# Combine all runs to delete
$runsToDelete = $debugRuns + $candidateCheckRuns + $agentTestRuns + $oldExactProofRuns + $old20260421Runs + $oldMiscRuns

Write-Host "DELETING:" -ForegroundColor Yellow
Write-Host "  Category 1: Debug/test runs ($($debugRuns.Count) runs)"
Write-Host "  Category 2: Candidate check runs ($($candidateCheckRuns.Count) runs)"
Write-Host "  Category 3: Agent test runs ($($agentTestRuns.Count) runs)"
Write-Host "  Category 4: Old exact proof trials ($($oldExactProofRuns.Count) runs)"
Write-Host "  Category 5: Old 20260421 trials ($($old20260421Runs.Count) runs)"
Write-Host "  Category 6: Old misc runs ($($oldMiscRuns.Count) runs)"
Write-Host "  TOTAL: $($runsToDelete.Count) runs"
Write-Host ""

# Calculate space to be freed
$totalSize = 0
$runsToDelete | ForEach-Object {
    $path = Join-Path $baseDir $_
    if (Test-Path $path) {
        $size = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $totalSize += $size
    }
}

Write-Host "Estimated space to free: $([math]::Round($totalSize / 1GB, 2)) GB" -ForegroundColor Cyan
Write-Host ""

# Prompt for confirmation
$confirmation = Read-Host "Proceed with deletion? (yes/no)"

if ($confirmation -eq "yes") {
    Write-Host ""
    Write-Host "Deleting runs..." -ForegroundColor Yellow
    
    $deletedCount = 0
    $failedCount = 0
    
    foreach ($run in $runsToDelete) {
        $path = Join-Path $baseDir $run
        if (Test-Path $path) {
            try {
                Remove-Item -Path $path -Recurse -Force -ErrorAction Stop
                Write-Host "  ✓ Deleted: $run" -ForegroundColor Green
                $deletedCount++
            }
            catch {
                Write-Host "  ✗ Failed to delete: $run - $($_.Exception.Message)" -ForegroundColor Red
                $failedCount++
            }
        }
        else {
            Write-Host "  - Skipped (not found): $run" -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "=== Cleanup Complete ===" -ForegroundColor Cyan
    Write-Host "Deleted: $deletedCount runs" -ForegroundColor Green
    Write-Host "Failed: $failedCount runs" -ForegroundColor Red
    Write-Host "Freed approximately: $([math]::Round($totalSize / 1GB, 2)) GB" -ForegroundColor Cyan
}
else {
    Write-Host ""
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
}
