// ui-tui/src/components/streaming.tsx
import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../theme';

interface StreamingProps {
  content: string;
  isActive: boolean;
}

const THINKING_STEPS = [
  'searching literature',
  'extracting entities',
  'building knowledge graph',
  'detecting anomalies',
  'generating mechanisms',
  'evaluating theories',
];

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

export const Streaming: React.FC<StreamingProps> = ({ content, isActive }) => {
  const [stepIdx, setStepIdx] = useState(0);
  const [spinIdx, setSpinIdx] = useState(0);

  useEffect(() => {
    if (!isActive) return;
    const interval = setInterval(() => {
      setSpinIdx(prev => (prev + 1) % SPINNER_FRAMES.length);
    }, 80);
    return () => clearInterval(interval);
  }, [isActive]);

  useEffect(() => {
    if (!isActive) return;
    const interval = setInterval(() => {
      setStepIdx(prev => (prev + 1) % THINKING_STEPS.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [isActive]);

  if (!isActive && !content) return null;

  if (isActive && !content) {
    return (
      <Box flexDirection="row" paddingLeft={1}>
        <Text color={theme.accent.cyan}>{SPINNER_FRAMES[spinIdx]} </Text>
        <Text color={theme.txt.secondary}>{THINKING_STEPS[stepIdx]}...</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingLeft={1}>
      <Text color={theme.txt.primary} wrap="wrap">{content}</Text>
    </Box>
  );
};
