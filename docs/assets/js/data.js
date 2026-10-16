/*
 * Numbers reported in the paper (Tables 1-2 and Figure 3).
 * SubPass@k values are recovered from the Figure 3 SVGs; their k = 1 points
 * match Table 1 exactly.
 */
(function (root) {
  'use strict';
  const AWM = (root.AWM = root.AWM || {});

  const PR_COLS = ['Easy', 'Medium', 'Difficult', 'Knowledge', 'Overall'];
  const PH_COLS = ['Stat. Mech.', 'Quantum', 'Atomic', 'E&M', 'Mechanics', 'Optics', 'Overall'];
  const METHODS = ['Vanilla', 'RLVR', 'Direct-Likert', 'Reference-Likert', 'Rubrics as Rewards', 'RLAWM'];

  // Table 1. Each row: 5 PhysReason columns, then 7 PHYSICS columns.
  const table1 = {
    'Qwen3-VL-8B': {
      'Vanilla': [39.5, 15.5, 13.3, 46.0, 30.0, 16.3, 17.3, 9.7, 33.8, 18.0, 16.7, 18.5],
      'RLVR': [39.1, 22.5, 17.5, 47.3, 32.8, 16.8, 16.3, 14.0, 25.0, 16.0, 36.1, 19.2],
      'Direct-Likert': [43.8, 25.7, 20.6, 52.8, 37.1, 23.7, 23.8, 14.4, 33.8, 19.7, 31.6, 23.8],
      'Reference-Likert': [31.0, 20.1, 20.3, 55.6, 33.2, 18.8, 22.3, 14.4, 31.6, 18.2, 34.2, 22.2],
      'Rubrics as Rewards': [38.7, 24.4, 19.5, 50.0, 34.4, 19.0, 23.0, 14.1, 30.7, 18.0, 32.3, 21.9],
      'RLAWM': [57.6, 39.6, 23.9, 64.0, 47.9, 26.4, 21.9, 28.0, 42.7, 20.5, 38.9, 29.0],
    },
    'Qwen3-VL-4B': {
      'Vanilla': [22.4, 19.6, 13.3, 52.7, 28.6, 10.3, 11.0, 9.0, 25.5, 3.9, 23.1, 13.1],
      'RLVR': [27.1, 22.0, 14.2, 52.0, 30.4, 16.4, 17.1, 9.8, 23.0, 16.0, 30.9, 17.7],
      'Direct-Likert': [31.4, 20.2, 9.7, 60.7, 32.6, 20.4, 8.5, 11.6, 31.8, 12.8, 27.8, 17.9],
      'Reference-Likert': [29.8, 26.0, 15.8, 50.7, 32.0, 19.9, 13.6, 8.2, 28.1, 17.4, 23.3, 17.8],
      'Rubrics as Rewards': [25.7, 23.1, 13.6, 46.0, 28.4, 17.3, 14.3, 9.8, 19.8, 19.4, 25.0, 16.7],
      'RLAWM': [51.5, 37.0, 23.1, 61.3, 44.8, 20.2, 19.6, 12.9, 31.3, 24.3, 32.2, 22.2],
    },
    'Gemma-3-12B': {
      'Vanilla': [11.4, 16.1, 11.1, 34.7, 19.3, 12.3, 3.5, 14.9, 25.1, 5.6, 22.2, 13.2],
      'RLVR': [15.9, 17.1, 14.1, 38.7, 22.5, 13.6, 9.3, 18.1, 21.1, 9.5, 19.1, 14.8],
      'Direct-Likert': [18.5, 15.5, 17.4, 37.4, 23.0, 18.3, 9.3, 13.6, 26.0, 11.3, 25.9, 16.7],
      'Reference-Likert': [17.7, 14.2, 16.3, 34.7, 21.5, 15.2, 8.8, 18.5, 22.9, 12.4, 21.4, 16.1],
      'Rubrics as Rewards': [15.6, 17.0, 13.9, 31.9, 20.4, 15.1, 7.6, 15.2, 22.6, 8.9, 21.7, 14.7],
      'RLAWM': [35.6, 19.4, 17.8, 44.9, 30.6, 19.7, 11.3, 15.4, 24.5, 14.9, 30.6, 18.3],
    },
  };

  // Figure 3, SubPass@k (%) on PhysReason.
  const passk = {
    ks: [1, 5, 10, 20, 50, 100],
    'Qwen3-VL-4B': {
      'Vanilla': [28.6, 48.6, 53.5, 59.8, 72.6, 76.5],
      'RLVR': [30.4, 49.9, 56.6, 60.0, 68.1, 72.6],
      'Reference-Likert': [32.0, 48.9, 56.6, 60.6, 70.1, 73.3],
      'RLAWM': [44.8, 62.4, 67.1, 71.0, 78.3, 81.1],
    },
    'Qwen3-VL-8B': {
      'Vanilla': [30.0, 54.4, 60.9, 67.8, 76.0, 80.1],
      'RLVR': [32.8, 53.8, 60.2, 66.6, 74.2, 78.8],
      'Reference-Likert': [33.2, 52.7, 59.6, 65.4, 72.9, 78.2],
      'RLAWM': [47.9, 60.5, 70.8, 73.7, 79.5, 83.4],
    },
  };

  // Table 2, zero-shot SubPass@1 (%).
  const ood = [
    { name: 'OlympiadBench-Phys', backbone: '4B', values: { 'RLVR': 4.5, 'Reference-Likert': 3.1, 'RLAWM': 11.3 } },
    { name: 'OlympiadBench-Phys', backbone: '8B', values: { 'RLVR': 6.2, 'Reference-Likert': 4.4, 'RLAWM': 15.2 } },
    { name: 'PHYBench', backbone: '8B', values: { 'RLVR': 5.8, 'Reference-Likert': 8.7, 'RLAWM': 10.5 } },
    { name: 'AIME 2026', backbone: '8B', values: { 'RLVR': 46.7, 'Reference-Likert': 43.3, 'RLAWM': 56.7 } },
  ];

  // Chart colours (validated as a set on the dark surface; Vanilla is a neutral reference).
  const COLORS = {
    'RLAWM': '#c98500',
    'Reference-Likert': '#3987e5',
    'RLVR': '#d55181',
    'Vanilla': '#8a93a6',
    baseline: '#a9b3c2',
  };

  AWM.data = { PR_COLS, PH_COLS, METHODS, table1, passk, ood, COLORS };
})(typeof self !== 'undefined' ? self : this);
