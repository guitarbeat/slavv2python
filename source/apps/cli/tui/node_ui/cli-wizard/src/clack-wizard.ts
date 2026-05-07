import * as p from '@clack/prompts';
import chalk from 'chalk';

export interface WizardConfig {
  projectName: string;
  environment: 'production' | 'staging' | 'development';
  modules: string[];
  outputDir: string;
  threadLimit: number;
}

export async function runClackWizard(): Promise<WizardConfig | null> {
  // Clear screen to ensure single-page feel
  console.clear();

  p.intro(chalk.bold.bgMagenta.white('  TERMICRAFT PIPELINE WIZARD  '));

  // 1. Project Name
  const projectName = await p.text({
    message: 'Enter the Project Name:',
    placeholder: 'my-vascular-pipeline',
    validate(value) {
      if (!value) return 'Project name cannot be empty!';
      if (/[^a-zA-Z0-9-_]/.test(value)) return 'Project name can only contain letters, numbers, dashes, and underscores!';
    },
  });

  if (p.isCancel(projectName)) {
    handleCancel();
    return null;
  }

  // 2. Select Environment
  const environment = await p.select({
    message: 'Select the Active Profile/Environment:',
    options: [
      { value: 'production', label: '🔥 Production', hint: 'Maximum optimization, full scale runs' },
      { value: 'staging', label: '⚡ Staging', hint: 'Balanced settings for pre-flight testing' },
      { value: 'development', label: '🛠️ Development', hint: 'Verbose debug reporting, parity assertions enabled' },
    ],
  });

  if (p.isCancel(environment)) {
    handleCancel();
    return null;
  }

  // 3. Multi-Select Modules (Spacebar navigation)
  const modules = await p.multiselect({
    message: 'Enable Pipeline Modules to execute (Press Space to toggle, Enter to confirm):',
    options: [
      { value: 'preprocessing', label: '1. Preprocessing & Contrast Stretching', hint: 'Required for high-noise TIFF volumes' },
      { value: 'feature_extraction', label: '2. Hessian Feature Extraction', hint: 'Extract tubular structures' },
      { value: 'edge_detection', label: '3. Global Watershed & Edge Curation', hint: 'Trace vessel outlines and seed points' },
      { value: 'vector_analysis', label: '4. Vectorization & Network Graphing', hint: 'Finalize nodes and export network.json' },
    ],
    required: true,
  });

  if (p.isCancel(modules)) {
    handleCancel();
    return null;
  }

  // 4. Output Directory
  const outputDir = await p.text({
    message: 'Enter Target Output Directory:',
    placeholder: './dev/runs/sample_a',
    initialValue: './slavv_output',
    validate(value) {
      if (!value) return 'Output directory cannot be empty!';
    },
  });

  if (p.isCancel(outputDir)) {
    handleCancel();
    return null;
  }

  // 5. Thread Limit (validated integer)
  const threadLimitStr = await p.text({
    message: 'Set Maximum CPU Thread Limit (1 - 32):',
    placeholder: '4',
    initialValue: '8',
    validate(value) {
      const parsed = parseInt(value, 10);
      if (isNaN(parsed) || parsed < 1 || parsed > 32) {
        return 'Thread limit must be a number between 1 and 32!';
      }
    },
  });

  if (p.isCancel(threadLimitStr)) {
    handleCancel();
    return null;
  }

  // 6. Confirm Transition
  const confirm = await p.confirm({
    message: 'Ready to launch the operations dashboard with these parameters?',
  });

  if (p.isCancel(confirm) || !confirm) {
    p.outro(chalk.yellow('Setup wizard aborted. Exiting.'));
    return null;
  }

  // Spinner animation for transition
  const s = p.spinner();
  s.start('Initializing Ink Operations Console...');
  await new Promise((resolve) => setTimeout(resolve, 1500));
  s.stop('Operations Console Initialized!');

  return {
    projectName: projectName as string,
    environment: environment as 'production' | 'staging' | 'development',
    modules: modules as string[],
    outputDir: outputDir as string,
    threadLimit: parseInt(threadLimitStr as string, 10),
  };
}

function handleCancel() {
  p.outro(chalk.red('Setup wizard cancelled. Exiting termicraft.'));
}