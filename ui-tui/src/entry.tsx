import React from 'react';
import { render, Box, Text } from 'ink';
import { App } from './app';

// Clear terminal and show fresh screen
process.stdout.write('\x1B[2J\x1B[3J\x1B[H\x1B[?25l');

render(<App />);
