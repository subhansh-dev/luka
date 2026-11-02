// ui-tui/src/components/banner.tsx
import React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../theme';

interface BannerProps {
  version: string;
  model: string;
}

const LOGO_LINES = [
  '██╗     ██╗   ██╗██╗  ██╗ █████╗ ',
  '██║     ██║   ██║██║ ██╔╝██╔══██╗',
  '██║     ██║   ██║█████╔╝ ███████║',
  '██║     ██║   ██║██╔═██╗ ██╔══██║',
  '███████╗╚██████╔╝██║  ██╗██║  ██║',
  '╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝',
];

export const Banner: React.FC<BannerProps> = ({ version, model }) => {
  return (
    <Box flexDirection="column" paddingX={1}>
      {/* Top border */}
      <Text color={theme.border.subtle}>
        {'─'.repeat(70)}
      </Text>

      {/* Logo + Info side by side */}
      <Box flexDirection="row">
        <Box flexDirection="column" paddingRight={2}>
          {LOGO_LINES.map((line, i) => (
            <Text key={i} color={theme.accent.cyan} bold>{line}</Text>
          ))}
        </Box>
        <Box flexDirection="column" justifyContent="center" paddingLeft={1}>
          <Text color={theme.txt.bright} bold>{'Autonomous Space Discovery Engine'}</Text>
          <Text color={theme.txt.muted}>{'  Built on RUMI Architecture'}</Text>
          <Box paddingTop={1} flexDirection="column">
            <Text color={theme.txt.secondary}>{`  v${version}  ·  ${model}  ·  8 LLM Providers`}</Text>
            <Text color={theme.txt.muted}>{'  29 Space APIs  ·  60+ Cognitive Modules'}</Text>
          </Box>
        </Box>
      </Box>

      {/* Bottom border */}
      <Text color={theme.border.subtle}>
        {'─'.repeat(70)}
      </Text>

      {/* Help hint */}
      <Box paddingTop={0} paddingBottom={0} paddingLeft={1}>
        <Text color={theme.txt.muted}>
          {'Type '}
          <Text color={theme.accent.cyan}>/help</Text>
          {' for commands  ·  '}
          <Text color={theme.accent.cyan}>/discover {'<topic>'}</Text>
          {'  ·  '}
          <Text color={theme.accent.cyan}>Ctrl+K</Text>
          {' palette'}
        </Text>
      </Box>
    </Box>
  );
};
