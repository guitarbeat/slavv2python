import React from 'react';
import { render } from 'ink';
import { runClackWizard } from './clack-wizard.js';
import { InkDashboard } from './ink-dashboard.js';

async function main() {
  // Phase 1: Run Clack Setup Wizard
  const config = await runClackWizard();
  
  if (!config) {
    // Graceful exit if cancelled
    process.exit(0);
  }

  // Phase 2: Launch Ink Operations Dashboard
  console.clear();
  const { waitUntilExit } = render(<InkDashboard config={config} />);
  
  // Wait until Ink terminates (e.g. on exit/quit command)
  await waitUntilExit();
  
  console.clear();
  console.log('✨ Termicraft CLI Operations completed successfully. Have a nice day! ✨\n');
}

main().catch((err) => {
  console.error('💥 Fatal execution error:', err);
  process.exit(1);
});
