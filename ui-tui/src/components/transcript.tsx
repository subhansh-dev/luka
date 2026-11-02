// ui-tui/src/components/transcript.tsx
import React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../theme';
import { TranscriptEntry } from '../hooks/useTranscript';

interface TranscriptProps {
  entries: TranscriptEntry[];
  maxHeight?: number;
}

export const Transcript: React.FC<TranscriptProps> = ({ entries, maxHeight = 100 }) => {
  const visible = entries.slice(-maxHeight);

  if (visible.length === 0) {
    return (
      <Box flexDirection="column" paddingLeft={1} paddingTop={1}>
        <Text color={theme.txt.muted}>{'  Start a conversation or run a discovery pipeline.'}</Text>
        <Text color={theme.txt.muted}>{'  Try: '}<Text color={theme.accent.cyan}>What is the Casimir effect?</Text></Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" flexShrink={1}>
      {visible.map(entry => (
        <TranscriptLine key={entry.id} entry={entry} />
      ))}
    </Box>
  );
};

const TranscriptLine: React.FC<{ entry: TranscriptEntry }> = ({ entry }) => {
  switch (entry.role) {
    case 'user':
      return (
        <Box flexDirection="column" paddingTop={1} paddingLeft={1}>
          <Box flexDirection="row">
            <Text color={theme.accent.blue} bold>{'❯ '}</Text>
            <Text color={theme.txt.bright} wrap="wrap">{entry.content}</Text>
          </Box>
        </Box>
      );

    case 'assistant':
      return (
        <Box flexDirection="column" paddingLeft={1} paddingBottom={0}>
          <Box flexDirection="row">
            <Text color={theme.accent.cyan}>{'◆ '}</Text>
            <Text color={theme.txt.primary} wrap="wrap">{entry.content}</Text>
          </Box>
        </Box>
      );

    case 'system':
      return (
        <Box paddingLeft={2} paddingY={0}>
          <Text color={theme.txt.muted} dimColor>{`· ${entry.content}`}</Text>
        </Box>
      );

    case 'error':
      return (
        <Box paddingLeft={1} paddingY={0}>
          <Text color={theme.accent.red}>{'✖ '}</Text>
          <Text color={theme.accent.red}>{entry.content}</Text>
        </Box>
      );

    default:
      return null;
  }
};
