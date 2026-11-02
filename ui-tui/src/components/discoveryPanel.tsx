// ui-tui/src/components/discoveryPanel.tsx
import React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../theme';

interface DiscoveryPanelProps {
  topic: string;
  progress: number;
  phase: string;
  papers: number;
  entities: number;
  edges: number;
  isActive: boolean;
}

const PIPELINE_STEPS = [
  'Literature',
  'Knowledge Graph',
  'Enrichment',
  'Gap Detection',
  'Anomalies',
  'Hidden Vars',
  'Mechanisms',
  'Predictions',
  'Theory Cup',
  'Peer Review',
  'Scoring',
  'Refinement',
];

export const DiscoveryPanel: React.FC<DiscoveryPanelProps> = ({
  topic, progress, phase, papers, entities, edges, isActive,
}) => {
  if (!isActive && progress === 0) return null;

  const completedSteps = Math.floor((progress / 100) * PIPELINE_STEPS.length);
  const currentStep = PIPELINE_STEPS.findIndex(s => s.toLowerCase().includes(phase.toLowerCase()));

  // Progress bar
  const barWidth = 40;
  const filled = Math.round((progress / 100) * barWidth);
  const bar = '█'.repeat(filled) + '░'.repeat(barWidth - filled);

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text color={theme.border.subtle}>{'─'.repeat(70)}</Text>

      {/* Header */}
      <Box flexDirection="row">
        <Text color={theme.accent.blue} bold>{'◆ Discovery'}</Text>
        <Text color={theme.txt.muted}>{'  '}</Text>
        <Text color={theme.accent.amber}>{phase || 'initializing'}</Text>
        <Text color={theme.txt.muted}>{'  '}</Text>
        <Text color={theme.txt.secondary}>{progress}%</Text>
      </Box>

      {/* Topic */}
      {topic && (
        <Box paddingLeft={2}>
          <Text color={theme.txt.secondary} wrap="wrap">{topic.length > 60 ? topic.slice(0, 60) + '...' : topic}</Text>
        </Box>
      )}

      {/* Progress bar */}
      <Box paddingLeft={2}>
        <Text color={progress > 70 ? theme.accent.green : progress > 30 ? theme.accent.amber : theme.accent.cyan}>
          {bar}
        </Text>
      </Box>

      {/* Pipeline steps in a compact row */}
      <Box paddingLeft={2} flexDirection="row" flexWrap="wrap">
        {PIPELINE_STEPS.map((step, i) => {
          const isDone = i < completedSteps || (currentStep >= 0 && i < currentStep);
          const isCurrent = i === currentStep || (currentStep < 0 && i === completedSteps);
          const color = isDone ? theme.accent.green : isCurrent ? theme.accent.cyan : theme.txt.muted;
          const icon = isDone ? '✓' : isCurrent ? '●' : '·';
          return (
            <Text key={i} color={color}>
              {icon}{step}{' '}
            </Text>
          );
        })}
      </Box>

      {/* Metrics */}
      <Box paddingLeft={2}>
        <Text color={theme.txt.muted}>
          {papers > 0 && <Text>{papers} papers</Text>}
          {entities > 0 && <Text>{'  ·  '}{entities} entities</Text>}
          {edges > 0 && <Text>{'  ·  '}{edges} relations</Text>}
        </Text>
      </Box>

      <Text color={theme.border.subtle}>{'─'.repeat(70)}</Text>
    </Box>
  );
};
