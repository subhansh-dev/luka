// ui-tui/src/components/statusBar.tsx
import React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../theme';

interface StatusBarProps {
  state: string;
  model: string;
  tokens: number;
  cost: number;
  uptime: number;
  thinkMode: boolean;
  diveMode: boolean;
  elapsed?: number;
  maxContext?: number;
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h${m}m`;
  if (m > 0) return `${m}m${s}s`;
  return `${s}s`;
}

function formatTokens(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m${s}s`;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  state, model, tokens, cost, uptime, thinkMode, diveMode,
  elapsed = 0,
}) => {
  const isReady = state === 'READY' || state === 'IDLE';
  const isBusy = state === 'THINKING' || state === 'PROCESSING';

  const stateIcon = isReady ? '●' : isBusy ? '◐' : '○';
  const stateColor = isReady ? theme.accent.green : isBusy ? theme.accent.amber : theme.txt.muted;
  const stateLabel = state === 'THINKING' ? 'thinking'
    : state === 'PROCESSING' ? 'processing'
    : state === 'SPEAKING' ? 'speaking'
    : state === 'LISTENING' ? 'connected'
    : 'ready';

  return (
    <Box flexDirection="column">
      <Text color={theme.border.subtle}>{'─'.repeat(70)}</Text>
      <Box flexDirection="row" paddingX={1}>
        <Text color={stateColor}>{stateIcon} </Text>
        <Text color={stateColor} bold>{stateLabel}</Text>
        <Text color={theme.txt.muted}>{'  '}</Text>
        <Text color={theme.txt.secondary}>{model}</Text>
        <Text color={theme.txt.muted}>{'  '}</Text>
        <Text color={theme.txt.muted}>{formatTokens(tokens)} tok</Text>
        <Text color={theme.txt.muted}>{'  '}</Text>
        <Text color={theme.txt.muted}>{formatUptime(uptime)}</Text>
        {elapsed > 0 && (
          <>
            <Text color={theme.txt.muted}>{'  '}</Text>
            <Text color={theme.accent.amber}>{formatElapsed(elapsed)}</Text>
          </>
        )}
        {cost > 0 && (
          <>
            <Text color={theme.txt.muted}>{'  '}</Text>
            <Text color={theme.accent.green}>${cost.toFixed(4)}</Text>
          </>
        )}
        {thinkMode && (
          <>
            <Text color={theme.txt.muted}>{'  '}</Text>
            <Text color={theme.accent.purple}>{'[think]'}</Text>
          </>
        )}
        {diveMode && (
          <>
            <Text color={theme.txt.muted}>{'  '}</Text>
            <Text color={theme.accent.cyan}>{'[dive]'}</Text>
          </>
        )}
      </Box>
    </Box>
  );
};
