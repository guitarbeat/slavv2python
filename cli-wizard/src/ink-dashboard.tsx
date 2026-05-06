import React, { useState, useEffect } from 'react';
import { Box, Text, useInput, useApp, useStdout } from 'ink';
import chalk from 'chalk';
import { WizardConfig } from './clack-wizard.js';

interface InkDashboardProps {
  config: WizardConfig;
}

const SLAVV_LOG_TEMPLATES = [
  'Loading 3D TIFF volume [volume.tif]...',
  'Volume dimensions detected: 512 x 512 x 128 (uint8)',
  'Preprocessing: Running contrast-limited adaptive histogram equalization (CLAHE)...',
  'Preprocessing: Normalizing voxel intensity range to [0.0, 1.0]...',
  'Feature Extraction: Calculating Hessian eigenvalues on scale 1.0...',
  'Feature Extraction: Hessian scale 2.0 processing completed successfully.',
  'Watershed: Tracing frontier pixels on global energy map...',
  'Watershed: Inserting 1,245,900 seeds into descending energy priority queue...',
  'Watershed: Expanding frontier. Energy order check passed.',
  'Watershed: Pixel expansion completed. 100% parity with MATLAB oracle achieved.',
  'Vectorization: Tracing 3D skeleton centerlines with medial axis thin...',
  'Vectorization: Pruning redundant cycles and resolving conflict junctions...',
  'Analysis: Successfully exported network.json (154 vertices, 192 edges).',
  'Plotting: Compiling standalone HTML visualization plots.html...',
];

export const InkDashboard: React.FC<InkDashboardProps> = ({ config }) => {
  const { exit } = useApp();
  const { stdout } = useStdout();

  // 1. Interactive input state
  const [inputText, setInputText] = useState('');
  const [showCursor, setShowCursor] = useState(true);

  // 2. Metrics state (live simulated data)
  const [cpuLoad, setCpuLoad] = useState(0);
  const [memUsage, setMemUsage] = useState(0);
  const [pipelineProgress, setPipelineProgress] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  // 3. Typewriter Log Stream state
  const [completedLogs, setCompletedLogs] = useState<string[]>([
    `[INFO] Workspace verified: c:\\Users\\alw4834\\Documents\\slavv2python`,
    `[INFO] Project [${config.projectName}] initialized successfully.`,
    `[INFO] Running on active profile: ${config.environment.toUpperCase()}`,
  ]);
  const [currentLogTemplateIndex, setCurrentLogTemplateIndex] = useState(0);
  const [currentTypedText, setCurrentTypedText] = useState('');
  const [currentCharIndex, setCurrentCharIndex] = useState(0);
  const [isWaitingForNextLog, setIsWaitingForNextLog] = useState(false);

  // Blinking cursor effect for command input
  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setShowCursor((prev) => !prev);
    }, 500);
    return () => clearInterval(cursorInterval);
  }, []);

  // Live metrics simulation
  useEffect(() => {
    if (isPaused) return;

    const metricsInterval = setInterval(() => {
      // Simulate CPU load fluctuation
      setCpuLoad(Math.floor(Math.random() * (92 - 64 + 1)) + 64);
      // Simulate Memory usage fluctuation
      setMemUsage(parseFloat((1.4 + Math.random() * 0.3).toFixed(2)));
      // Increment pipeline progress
      setPipelineProgress((prev) => {
        if (prev >= 100) return 0;
        return prev + 1;
      });
    }, 1000);

    return () => clearInterval(metricsInterval);
  }, [isPaused]);

  // Typewriter Log Stream Simulation
  useEffect(() => {
    if (isPaused) return;

    const targetText = SLAVV_LOG_TEMPLATES[currentLogTemplateIndex];

    if (!isWaitingForNextLog) {
      if (currentCharIndex < targetText.length) {
        const typingTimeout = setTimeout(() => {
          setCurrentTypedText((prev) => prev + targetText[currentCharIndex]);
          setCurrentCharIndex((prev) => prev + 1);
        }, 25);
        return () => clearTimeout(typingTimeout);
      } else {
        // Line finished typing, set waiting state
        setIsWaitingForNextLog(true);
      }
    } else {
      // Waiting state: delay before pushing to completed and starting next line
      const nextLineTimeout = setTimeout(() => {
        setCompletedLogs((prev) => [...prev.slice(-15), `[INFO] ${targetText}`]);
        setCurrentLogTemplateIndex((prev) => (prev + 1) % SLAVV_LOG_TEMPLATES.length);
        setCurrentTypedText('');
        setCurrentCharIndex(0);
        setIsWaitingForNextLog(false);
      }, 2000);
      return () => clearTimeout(nextLineTimeout);
    }
  }, [currentLogTemplateIndex, currentCharIndex, isPaused, isWaitingForNextLog]);

  // Handle Command Submission
  const handleCommand = (cmd: string) => {
    const trimmed = cmd.trim().toLowerCase();
    if (trimmed === 'exit' || trimmed === 'quit') {
      setCompletedLogs((prev) => [...prev, `[CMD] ${cmd}`, '[SYS] Exiting dashboard. Goodbye!']);
      setTimeout(() => {
        exit();
      }, 500);
    } else if (trimmed === 'pause') {
      setIsPaused(true);
      setCompletedLogs((prev) => [...prev, `[CMD] ${cmd}`, '[SYS] Simulation and typewriter stream PAUSED.']);
    } else if (trimmed === 'resume') {
      setIsPaused(false);
      setCompletedLogs((prev) => [...prev, `[CMD] ${cmd}`, '[SYS] Simulation and typewriter stream RESUMED.']);
    } else if (trimmed === 'clear') {
      setCompletedLogs([]);
      setCurrentTypedText('');
      setCurrentCharIndex(0);
      setIsWaitingForNextLog(false);
    } else if (trimmed === 'help') {
      setCompletedLogs((prev) => [
        ...prev,
        `[CMD] ${cmd}`,
        '[HELP] Available Commands:',
        '  • pause   - Pause live logs & metric simulation',
        '  • resume  - Resume live logs & metric simulation',
        '  • clear   - Clear the log terminal window',
        '  • exit    - Shut down the Operations Console and exit',
      ]);
    } else if (trimmed) {
      setCompletedLogs((prev) => [
        ...prev,
        `[CMD] ${cmd}`,
        `[WARN] Unknown command: "${cmd}". Type "help" for a list of available commands.`,
      ]);
    }
  };

  // Capture user input
  useInput((input, key) => {
    if (key.return) {
      handleCommand(inputText);
      setInputText('');
    } else if (key.backspace) {
      setInputText(inputText.slice(0, -1));
    } else if (
      input &&
      !key.ctrl &&
      !key.meta &&
      !key.upArrow &&
      !key.downArrow &&
      !key.leftArrow &&
      !key.rightArrow &&
      !key.escape &&
      !key.pageUp &&
      !key.pageDown
    ) {
      // Filter out non-printable/control characters that might slip through
      if (input.charCodeAt(0) >= 32 || input === '\t') {
        setInputText(inputText + input);
      }
    }
  });

  // Custom Progress Bar Builder
  const renderProgressBar = (percentage: number) => {
    const totalBars = 20;
    const filledBars = Math.round((percentage / 100) * totalBars);
    const emptyBars = totalBars - filledBars;
    const filledStr = '█'.repeat(filledBars);
    const emptyStr = '░'.repeat(emptyBars);
    return `[${filledStr}${emptyStr}] ${percentage}%`;
  };

  const terminalHeight = stdout?.rows || 24;

  return (
    <Box flexDirection="column" width="100%" height={terminalHeight} paddingX={1}>
      {/* HEADER SECTION */}
      <Box
        borderStyle="double"
        borderColor="magenta"
        width="100%"
        justifyContent="center"
        paddingX={2}
        marginBottom={1}
      >
        <Text bold color="cyan">
          🚀 TERMICRAFT OPERATIONS CONSOLE •{' '}
          <Text color="yellow">{config.projectName.toUpperCase()}</Text>
        </Text>
      </Box>

      {/* BODY COLUMN SECTION */}
      <Box flexDirection="row" width="100%" flexGrow={1} marginBottom={1}>
        {/* LEFT COLUMN: METRICS & DETAILS (40% width) */}
        <Box
          borderStyle="round"
          borderColor="blue"
          width="40%"
          flexDirection="column"
          paddingX={1}
          marginRight={1}
        >
          <Text bold color="yellow" underline>
            ⚙️ SYSTEM PROFILE
          </Text>
          <Box flexDirection="column" marginY={1}>
            <Text>
              Profile: <Text color="green" bold>{config.environment.toUpperCase()}</Text>
            </Text>
            <Text>
              Threads: <Text color="green" bold>{config.threadLimit} Cores</Text>
            </Text>
            <Text ellipsizeMode="end">
              Output: <Text color="gray">{config.outputDir.length > 22 ? config.outputDir.slice(0, 19) + '...' : config.outputDir}</Text>
            </Text>
          </Box>

          <Text bold color="yellow" underline>
            📦 ENABLED MODULES
          </Text>
          <Box flexDirection="column" marginY={1}>
            {config.modules.map((m) => {
              const labelMap: Record<string, string> = {
                preprocessing: '• Preprocessing & CLAHE',
                feature_extraction: '• Hessian Feature Extract',
                edge_detection: '• Global Watershed Grid',
                vector_analysis: '• Skeleton & Vector Graph',
              };
              return (
                <Text key={m} color="cyan">
                  {labelMap[m] || `• ${m}`}
                </Text>
              );
            })}
          </Box>

          <Text bold color="yellow" underline>
            📈 LIVE METRICS
          </Text>
          <Box flexDirection="column" marginTop={1}>
            <Text>
              CPU Load: <Text color={cpuLoad > 85 ? 'red' : 'green'} bold>{cpuLoad}%</Text>
            </Text>
            <Text>
              Memory: <Text color="green" bold>{memUsage} GB / 32 GB</Text>
            </Text>
            <Box flexDirection="column" marginTop={1}>
              <Text>Progress:</Text>
              <Text color="green" bold>
                {renderProgressBar(pipelineProgress)}
              </Text>
            </Box>
          </Box>
        </Box>

        {/* RIGHT COLUMN: RUNNING LOGS (60% width) */}
        <Box
          borderStyle="round"
          borderColor="blue"
          width="60%"
          flexDirection="column"
          paddingX={1}
        >
          <Text bold color="green" underline>
            📋 PIPELINE LOG OPERATIONS
          </Text>
          <Box flexDirection="column" flexGrow={1} marginY={1}>
            {completedLogs.map((log, index) => {
              let logColor = 'white';
              if (log.startsWith('[CMD]')) logColor = 'cyan';
              else if (log.startsWith('[SYS]')) logColor = 'yellow';
              else if (log.startsWith('[HELP]')) logColor = 'blue';
              else if (log.startsWith('[WARN]')) logColor = 'red';
              else if (log.includes('100% parity')) logColor = 'green';

              return (
                <Text key={index} color={logColor}>
                  {log}
                </Text>
              );
            })}
            {/* The active typewriter stream log */}
            {currentTypedText && (
              <Text color="white" bold>
                ⏱️ [INFO] {currentTypedText}
                <Text color="cyan">{showCursor ? '▮' : ' '}</Text>
              </Text>
            )}
          </Box>
          <Box justifyContent="flex-end">
            <Text color="gray" italic>
              Status: {isPaused ? chalk.yellow('⏸️ PAUSED') : chalk.green('▶️ RUNNING')}
            </Text>
          </Box>
        </Box>
      </Box>

      {/* FOOTER INPUT BOX */}
      <Box borderStyle="single" borderColor="cyan" width="100%" paddingX={1}>
        <Text bold color="cyan">
          {'> '}
        </Text>
        <Text color="white">
          {inputText}
          {showCursor && <Text color="cyan">▮</Text>}
        </Text>
        {!inputText && (
          <Text color="gray" italic>
            {' '}Type command (help, pause, resume, clear, exit)...
          </Text>
        )}
      </Box>
    </Box>
  );
};
