/**
 * LLM Pipeline Explorer - Core JavaScript Application
 * Vanilla HTML5, CSS3, & JavaScript (No external libraries)
 *
 * Pipeline Flow:
 * Corpus -> Vocabulary (Token + ID) -> Sentence Tokenization -> Token IDs -> Embeddings -> 3D Canvas
 */

(function () {
  'use strict';

  // =========================================================================
  // State Management
  // =========================================================================
  const state = {
    // Section 1: Vocabulary State
    vocabulary: new Map(), // Map<normalizedString, { id, display, count, type }>
    vocabSize: 0,
    totalCorpusTokens: 0,
    isVocabBuilt: false,

    // Section 2: Tokenization State
    lastTokenizedSentence: '',
    tokens: [],

    // Section 3: 3D Visualization State
    mode: 'spelling', // 'spelling' | 'meaning'
    sentences: [],
    points3D: [], // Array<{ x, y, z, sentence, index, category, screenX, screenY, depth }>
    hoveredIndex: -1,

    // 3D Camera Controls
    camera: {
      rotX: -0.32, // Pitch (radians)
      rotY: 0.65,  // Yaw (radians)
      zoom: 1.0,
      baseDist: 500,
      isDragging: false,
      lastMouseX: 0,
      lastMouseY: 0
    }
  };

  let prefersReducedMotion = false;

  // =========================================================================
  // DOM Elements
  // =========================================================================
  const elements = {
    // Flow Tracker
    flowSteps: [
      document.getElementById('flow-step-1'),
      document.getElementById('flow-step-2'),
      document.getElementById('flow-step-3'),
      document.getElementById('flow-step-4'),
      document.getElementById('flow-step-5'),
      document.getElementById('flow-step-6')
    ],

    // Section 1: Vocab
    corpusInput: document.getElementById('corpus-input'),
    btnBuildVocab: document.getElementById('btn-build-vocab'),
    vocabStatus: document.getElementById('vocab-status'),
    vocabResultsArea: document.getElementById('vocab-results-area'),
    statVocabSize: document.getElementById('stat-vocab-size'),
    statTotalTokens: document.getElementById('stat-total-tokens'),
    vocabSearch: document.getElementById('vocab-search'),
    vocabFilterCount: document.getElementById('vocab-filter-count'),
    vocabTableBody: document.getElementById('vocab-table-body'),

    // Section 2: Tokenize
    sentenceInput: document.getElementById('sentence-input'),
    btnTokenize: document.getElementById('btn-tokenize'),
    tokenizeWarning: document.getElementById('tokenize-vocab-warning'),
    tokenizeResultsArea: document.getElementById('tokenize-results-area'),
    tokenCountValue: document.getElementById('token-count-value'),
    knownCountValue: document.getElementById('known-count-value'),
    unknownCountValue: document.getElementById('unknown-count-value'),
    tokensChipList: document.getElementById('tokens-chip-list'),
    tokenIdStream: document.getElementById('token-id-stream'),

    // Section 3: 3D Embeddings
    embeddingsSentences: document.getElementById('embeddings-sentences'),
    modeSpelling: document.getElementById('mode-spelling'),
    modeMeaning: document.getElementById('mode-meaning'),
    btnPlot3D: document.getElementById('btn-plot-3d'),
    btnResetView: document.getElementById('btn-reset-view'),
    btnZoomIn: document.getElementById('btn-zoom-in'),
    btnZoomOut: document.getElementById('btn-zoom-out'),
    activeModeIndicator: document.getElementById('mode-name-label'),
    heuristicDetailsText: document.getElementById('heuristic-details-text'),
    canvas: document.getElementById('scatter-canvas'),
    canvasContainer: document.getElementById('canvas-container'),
    pointTooltip: document.getElementById('point-tooltip')
  };

  // Canvas 2D Rendering Context
  const ctx = elements.canvas.getContext('2d');

  // Palette constants matching CSS tokens
  const THEME_COLORS = {
    feline: '#047857',   // Emerald
    lexical: '#b45309',  // Amber
    sky: '#1d4ed8',      // Blue
    tech: '#0f766e'      // Teal (Cohesive palette, no AI violet slop)
  };

  // =========================================================================
  // SECTION 1: VOCABULARY BUILDER
  // =========================================================================

  /**
   * Splits raw text into an array of lexical units (words and punctuation).
   * Whitespace is ignored; punctuation marks are isolated as distinct tokens.
   */
  function extractRawTokens(text) {
    if (!text || typeof text !== 'string') return [];
    const regex = /([a-zA-Z0-9]+|[^\s\w])/g;
    return text.match(regex) || [];
  }

  /**
   * Builds vocabulary from user corpus text.
   * - Reserved token ID 0 is '[UNK]'.
   * - Corpus tokens are normalized for matching (lowercase) while preserving display text.
   * - Deterministic IDs are assigned starting from 1 in order of appearance.
   */
  function buildVocabulary() {
    const rawText = elements.corpusInput.value.trim();
    if (!rawText) {
      alert('Please enter at least one sentence in the corpus text box.');
      return;
    }

    state.vocabulary.clear();

    // Reserve ID 0 for unknown tokens
    state.vocabulary.set('[unk]', {
      id: 0,
      display: '[UNK]',
      count: 0,
      type: 'special'
    });

    const rawTokens = extractRawTokens(rawText);
    state.totalCorpusTokens = rawTokens.length;

    let nextId = 1;
    for (const raw of rawTokens) {
      const normalized = raw.toLowerCase();
      const isPunctuation = /^[^\s\w]$/.test(raw);

      if (state.vocabulary.has(normalized)) {
        state.vocabulary.get(normalized).count++;
      } else {
        state.vocabulary.set(normalized, {
          id: nextId++,
          display: raw,
          count: 1,
          type: isPunctuation ? 'punctuation' : 'word'
        });
      }
    }

    state.vocabSize = state.vocabulary.size;
    state.isVocabBuilt = true;

    // Update UI status
    elements.vocabStatus.textContent = 'Vocabulary built successfully';
    elements.vocabStatus.className = 'status-indicator ready';
    elements.statVocabSize.textContent = state.vocabSize.toString();
    elements.statTotalTokens.textContent = state.totalCorpusTokens.toString();
    elements.vocabResultsArea.classList.remove('hidden');

    // Hide any previous warning in Section 2
    elements.tokenizeWarning.classList.add('hidden');

    // Render table
    renderVocabularyTable();

    // GSAP Stat & Table Reveal
    if (typeof gsap !== 'undefined' && !prefersReducedMotion) {
      gsap.fromTo(
        ['#stat-vocab-size', '#stat-total-tokens'],
        { scale: 0.82, autoAlpha: 0 },
        { scale: 1, autoAlpha: 1, duration: 0.45, ease: 'back.out(1.7)' }
      );
      gsap.fromTo(
        '#vocab-table-body tr',
        { y: 8, autoAlpha: 0 },
        { y: 0, autoAlpha: 1, duration: 0.3, stagger: 0.015, ease: 'power1.out' }
      );
    }

    // Invalidate stale tokenization results from Section 2
    invalidateTokenizationResults();

    // Update pipeline flow steps
    updateFlowStep(2, true);
  }

  /**
   * Renders the vocabulary table with optional search filter.
   */
  function renderVocabularyTable() {
    const searchTerm = elements.vocabSearch.value.trim().toLowerCase();
    const rows = [];
    let matchCount = 0;

    const vocabList = Array.from(state.vocabulary.values()).sort((a, b) => a.id - b.id);

    for (const item of vocabList) {
      if (searchTerm && !item.display.toLowerCase().includes(searchTerm)) {
        continue;
      }
      matchCount++;

      const isUnk = item.id === 0;
      const typeClass = item.type === 'word' ? 'type-word' : item.type === 'punctuation' ? 'type-punct' : 'type-special';
      const typeLabel = item.type === 'word' ? 'Word' : item.type === 'punctuation' ? 'Punctuation' : 'Special';

      rows.push(`
        <tr>
          <td><span class="token-id-badge ${isUnk ? 'unk-badge' : ''}">${item.id}</span></td>
          <td><strong>${escapeHtml(item.display)}</strong></td>
          <td><span class="type-pill ${typeClass}">${typeLabel}</span></td>
          <td>${isUnk ? '—' : item.count}</td>
        </tr>
      `);
    }

    elements.vocabTableBody.innerHTML = rows.join('');
    elements.vocabFilterCount.textContent = searchTerm
      ? `Showing ${matchCount} of ${vocabList.length} items`
      : `${vocabList.length} unique tokens`;
  }

  // =========================================================================
  // SECTION 2: SENTENCE TOKENIZER
  // =========================================================================

  /**
   * Tokenizes an input sentence using the built vocabulary.
   */
  function tokenizeSentence() {
    if (!state.isVocabBuilt) {
      elements.tokenizeWarning.classList.remove('hidden');
      elements.tokenizeResultsArea.classList.add('hidden');
      return;
    }

    elements.tokenizeWarning.classList.add('hidden');
    const inputSentence = elements.sentenceInput.value.trim();
    if (!inputSentence) {
      alert('Please enter a sentence to tokenize.');
      return;
    }

    state.lastTokenizedSentence = inputSentence;
    const rawTokens = extractRawTokens(inputSentence);
    const resultTokens = [];

    for (const raw of rawTokens) {
      const normalized = raw.toLowerCase();
      const isPunctuation = /^[^\s\w]$/.test(raw);

      if (state.vocabulary.has(normalized)) {
        const vocabItem = state.vocabulary.get(normalized);
        resultTokens.push({
          text: raw,
          id: vocabItem.id,
          isUnknown: false,
          isSubword: false
        });
      } else {
        if (isPunctuation) {
          resultTokens.push({
            text: raw,
            id: 0,
            isUnknown: true,
            isSubword: false
          });
        } else {
          const chars = raw.split('');
          for (let i = 0; i < chars.length; i++) {
            const charToken = i === 0 ? chars[i] : `##${chars[i]}`;
            resultTokens.push({
              text: charToken,
              id: 0,
              isUnknown: true,
              isSubword: i > 0
            });
          }
        }
      }
    }

    state.tokens = resultTokens;
    renderTokenizationResults();
    updateFlowStep(4, true);
  }

  /**
   * Renders token chips and raw ID stream.
   */
  function renderTokenizationResults() {
    const totalCount = state.tokens.length;
    const knownCount = state.tokens.filter(t => !t.isUnknown).length;
    const unknownCount = totalCount - knownCount;

    elements.tokenCountValue.textContent = totalCount.toString();
    elements.knownCountValue.textContent = knownCount.toString();
    elements.unknownCountValue.textContent = unknownCount.toString();

    const chipsHtml = state.tokens.map(token => `
      <div class="token-chip ${token.isUnknown ? 'chip-unknown' : ''}" title="Token: ${escapeHtml(token.text)} (ID: ${token.id})">
        <span class="chip-text">${escapeHtml(token.text)}</span>
        <div class="chip-meta">
          <span class="chip-id ${token.id === 0 ? 'chip-id-zero' : ''}">ID ${token.id}</span>
          ${token.isUnknown ? '<span class="chip-label-unk">[UNK]</span>' : ''}
        </div>
      </div>
    `).join('');

    elements.tokensChipList.innerHTML = chipsHtml;

    const idArray = state.tokens.map(t => t.id);
    elements.tokenIdStream.textContent = `[ ${idArray.join(', ')} ]`;

    elements.tokenizeResultsArea.classList.remove('hidden');

    // GSAP Staggered Entrance for Token Chips and Stream
    if (typeof gsap !== 'undefined' && !prefersReducedMotion) {
      gsap.fromTo(
        '.token-chip',
        { scale: 0.75, y: -10, autoAlpha: 0 },
        { scale: 1, y: 0, autoAlpha: 1, duration: 0.35, stagger: 0.035, ease: 'back.out(1.6)' }
      );
      gsap.fromTo(
        '#token-id-stream',
        { autoAlpha: 0, x: -14 },
        { autoAlpha: 1, x: 0, duration: 0.4, ease: 'power2.out' }
      );
    }
  }

  function invalidateTokenizationResults() {
    state.tokens = [];
    state.lastTokenizedSentence = '';
    elements.tokenizeResultsArea.classList.add('hidden');
    elements.tokenizeWarning.classList.add('hidden');
  }

  // =========================================================================
  // SECTION 3: SENTENCE EMBEDDINGS & 3D VISUALIZATION
  // =========================================================================

  const CONCEPT_KEYWORDS = {
    feline: ['cat', 'kitten', 'feline', 'mat', 'rug', 'carpet', 'sat', 'rests', 'sitting', 'relaxes', 'rest', 'nap'],
    celestial: ['sun', 'solar', 'shining', 'rays', 'illuminate', 'brightly', 'bright', 'clear', 'sky', 'atmosphere', 'cloudless', 'daylight'],
    tech: ['quantum', 'computers', 'computer', 'processors', 'processor', 'calculate', 'algorithms', 'algorithm', 'complex', 'mathematical', 'numerical', 'problems', 'solve', 'advanced', 'compute'],
    bat: ['bat', 'wings', 'flying', 'nocturnal', 'cave']
  };

  function levenshteinDistance(s1, s2) {
    const a = s1.toLowerCase();
    const b = s2.toLowerCase();
    const m = a.length;
    const n = b.length;
    if (m === 0) return n;
    if (n === 0) return m;

    const matrix = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));
    for (let i = 0; i <= m; i++) matrix[i][0] = i;
    for (let j = 0; j <= n; j++) matrix[0][j] = j;

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;
        matrix[i][j] = Math.min(
          matrix[i - 1][j] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j - 1] + cost
        );
      }
    }
    return matrix[m][n];
  }

  function getCharacterNgrams(str, n = 2) {
    const s = str.toLowerCase().replace(/\s+/g, ' ');
    const ngrams = new Set();
    for (let i = 0; i <= s.length - n; i++) {
      ngrams.add(s.substring(i, i + n));
    }
    return ngrams;
  }

  function jaccardDistance(setA, setB) {
    if (setA.size === 0 && setB.size === 0) return 0;
    let intersection = 0;
    for (const item of setA) {
      if (setB.has(item)) intersection++;
    }
    const union = setA.size + setB.size - intersection;
    return union === 0 ? 0 : 1 - (intersection / union);
  }

  function getSemanticVector(sentence) {
    const words = extractRawTokens(sentence).map(w => w.toLowerCase());
    const vec = [0, 0, 0, 0];

    for (const word of words) {
      if (CONCEPT_KEYWORDS.feline.includes(word)) vec[0] += 1.5;
      if (CONCEPT_KEYWORDS.celestial.includes(word)) vec[1] += 1.5;
      if (CONCEPT_KEYWORDS.tech.includes(word)) vec[2] += 1.5;
      if (CONCEPT_KEYWORDS.bat.includes(word)) vec[3] += 2.0;
    }

    vec[0] += (sentence.length % 5) * 0.05;
    vec[1] += (words.length % 4) * 0.05;
    vec[2] += (sentence.charCodeAt(0) % 3) * 0.05;

    const norm = Math.hypot(...vec) || 1;
    return vec.map(v => v / norm);
  }

  function cosineDistance(v1, v2) {
    let dot = 0;
    for (let i = 0; i < v1.length; i++) {
      dot += v1[i] * v2[i];
    }
    return Math.max(0, 1 - dot);
  }

  function classicalMDS(distMatrix, dimensions = 3) {
    const n = distMatrix.length;
    const D2 = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        D2[i][j] = distMatrix[i][j] * distMatrix[i][j];
      }
    }

    const rowMeans = new Float64Array(n);
    const colMeans = new Float64Array(n);
    let totalMean = 0;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        rowMeans[i] += D2[i][j];
        colMeans[j] += D2[i][j];
        totalMean += D2[i][j];
      }
    }
    for (let i = 0; i < n; i++) rowMeans[i] /= n;
    for (let j = 0; j < n; j++) colMeans[j] /= n;
    totalMean /= (n * n);

    const B = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        B[i][j] = -0.5 * (D2[i][j] - rowMeans[i] - colMeans[j] + totalMean);
      }
    }

    const coords = Array.from({ length: n }, () => new Float64Array(dimensions));
    const eigenvectors = [];

    for (let d = 0; d < dimensions; d++) {
      let vec = new Float64Array(n);
      for (let i = 0; i < n; i++) vec[i] = Math.sin(i * 1.5 + d + 1.0);

      for (let iter = 0; iter < 30; iter++) {
        for (const prevVec of eigenvectors) {
          let dot = 0;
          for (let i = 0; i < n; i++) dot += vec[i] * prevVec[i];
          for (let i = 0; i < n; i++) vec[i] -= dot * prevVec[i];
        }

        const next = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let j = 0; j < n; j++) sum += B[i][j] * vec[j];
          next[i] = sum;
        }

        const norm = Math.hypot(...next);
        if (norm > 1e-9) {
          for (let i = 0; i < n; i++) vec[i] = next[i] / norm;
        }
      }

      let lambda = 0;
      for (let i = 0; i < n; i++) {
        let bRowDot = 0;
        for (let j = 0; j < n; j++) bRowDot += B[i][j] * vec[j];
        lambda += vec[i] * bRowDot;
      }

      eigenvectors.push(vec);
      const scale = Math.sqrt(Math.max(0, lambda));
      for (let i = 0; i < n; i++) {
        coords[i][d] = vec[i] * scale;
      }
    }

    let maxDist = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(coords[i][0], coords[i][1], coords[i][2]);
      if (d > maxDist) maxDist = d;
    }
    const targetScale = maxDist > 1e-4 ? 130 / maxDist : 100;
    return coords.map(c => [c[0] * targetScale, c[1] * targetScale, c[2] * targetScale]);
  }

  function categorizeSentence(sentence, index) {
    const s = sentence.toLowerCase();
    if (s.includes('cat sat on the ...') || s.includes('bat sat on the ...')) {
      return 'lexical';
    }
    if (s.includes('cat') || s.includes('kitten') || s.includes('feline') || s.includes('mat') || s.includes('rug') || s.includes('carpet')) {
      return 'feline';
    }
    if (s.includes('sun') || s.includes('solar') || s.includes('sky') || s.includes('rays') || s.includes('atmosphere')) {
      return 'sky';
    }
    if (s.includes('quantum') || s.includes('computer') || s.includes('processor') || s.includes('algorithm')) {
      return 'tech';
    }
    return index % 2 === 0 ? 'feline' : 'lexical';
  }

  function calculateEmbeddings() {
    const rawLines = elements.embeddingsSentences.value
      .split('\n')
      .map(s => s.trim())
      .filter(s => s.length > 0);

    if (rawLines.length === 0) {
      alert('Please enter at least two sentences to plot.');
      return;
    }

    state.sentences = rawLines;
    const n = rawLines.length;
    const distMatrix = Array.from({ length: n }, () => new Float64Array(n));

    if (state.mode === 'spelling') {
      const ngramsList = rawLines.map(s => getCharacterNgrams(s, 2));

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) {
            distMatrix[i][j] = 0;
            continue;
          }
          const maxLen = Math.max(rawLines[i].length, rawLines[j].length) || 1;
          const levNorm = levenshteinDistance(rawLines[i], rawLines[j]) / maxLen;
          const jacDist = jaccardDistance(ngramsList[i], ngramsList[j]);
          distMatrix[i][j] = 0.5 * levNorm + 0.5 * jacDist;
        }
      }

      elements.activeModeIndicator.textContent = 'Spelling-based (static)';
      elements.heuristicDetailsText.innerHTML = `
        <strong>Spelling-based (static):</strong> Coordinates come from character n-gram Jaccard overlap and Levenshtein edit distance. Sentences <em>“The cat sat on the ...”</em> (#3) and <em>“The bat sat on the ...”</em> (#6) land together because their character spelling is 95% identical, even though a cat and a bat have nothing in common.
      `;
    } else {
      const semanticVectors = rawLines.map(s => getSemanticVector(s));

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) {
            distMatrix[i][j] = 0;
            continue;
          }
          distMatrix[i][j] = cosineDistance(semanticVectors[i], semanticVectors[j]);
        }
      }

      elements.activeModeIndicator.textContent = 'Meaning-based (contextual)';
      elements.heuristicDetailsText.innerHTML = `
        <strong>Meaning-based (contextual):</strong> Coordinates come from semantic topic vectors. Sentences <em>“The cat sat on the mat.”</em> (#1) and <em>“Kitten rests on the rug.”</em> (#2) cluster together because their topic vectors match, despite sharing zero words. Sentence #6 (bat) separates from the cluster because its context diverges.
      `;
    }

    const coords3D = classicalMDS(distMatrix, 3);
    const hasExisting = state.points3D && state.points3D.length === rawLines.length;

    if (hasExisting && typeof gsap !== 'undefined' && !prefersReducedMotion) {
      // Kill any in-flight tweens on point coordinates
      state.points3D.forEach(pt => gsap.killTweensOf(pt));

      rawLines.forEach((sentence, idx) => {
        const pt = state.points3D[idx];
        const [targetX, targetY, targetZ] = coords3D[idx];
        pt.sentence = sentence;
        pt.category = categorizeSentence(sentence, idx);

        gsap.to(pt, {
          x: targetX,
          y: targetY,
          z: targetZ,
          duration: 0.85,
          delay: idx * 0.02,
          ease: 'power2.out',
          onUpdate: renderCanvas3D
        });
      });
    } else {
      state.points3D = rawLines.map((sentence, idx) => {
        const [x, y, z] = coords3D[idx];
        return {
          sentence,
          index: idx + 1,
          x,
          y,
          z,
          category: categorizeSentence(sentence, idx),
          screenX: 0,
          screenY: 0,
          depth: 0
        };
      });
      renderCanvas3D();
    }
    updateFlowStep(6, true);
  }

  // =========================================================================
  // 3D CANVAS RENDERING ENGINE
  // =========================================================================

  function renderCanvas3D() {
    const width = elements.canvas.width;
    const height = elements.canvas.height;
    ctx.clearRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const { rotX, rotY, zoom, baseDist } = state.camera;

    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);

    function project3D(x, y, z) {
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;

      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;

      const cameraZ = baseDist + z2;
      const factor = (baseDist / Math.max(cameraZ, 40)) * zoom;

      return {
        x: centerX + x1 * factor,
        y: centerY + y2 * factor,
        z: z2,
        scale: factor
      };
    }

    // 1. Draw 3D Ground Grid (Floor at Y = 100)
    const floorY = 100;
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    const gridSize = 140;
    const gridStep = 35;

    ctx.beginPath();
    for (let g = -gridSize; g <= gridSize; g += gridStep) {
      const p1 = project3D(-gridSize, floorY, g);
      const p2 = project3D(gridSize, floorY, g);
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);

      const p3 = project3D(g, floorY, -gridSize);
      const p4 = project3D(g, floorY, gridSize);
      ctx.moveTo(p3.x, p3.y);
      ctx.lineTo(p4.x, p4.y);
    }
    ctx.stroke();

    // 2. Draw 3D Coordinate Axes from origin (0, 0, 0)
    const origin = project3D(0, 0, 0);
    const axisLen = 130;

    const axisX = project3D(axisLen, 0, 0);
    const axisY = project3D(0, -axisLen, 0);
    const axisZ = project3D(0, 0, axisLen);

    drawAxisLine(origin, axisX, '#dc2626', 'X (Dim 1)');
    drawAxisLine(origin, axisY, '#059669', 'Y (Dim 2)');
    drawAxisLine(origin, axisZ, '#1d4ed8', 'Z (Dim 3)');

    // 3. Project all 3D sentence points
    for (const pt of state.points3D) {
      const proj = project3D(pt.x, pt.y, pt.z);
      pt.screenX = proj.x;
      pt.screenY = proj.y;
      pt.depth = proj.z;
      pt.scale = proj.scale;
    }

    const sortedPoints = [...state.points3D].sort((a, b) => b.depth - a.depth);

    // 4. Draw Drop-lines to floor grid for depth perception
    for (const pt of sortedPoints) {
      const floorProj = project3D(pt.x, floorY, pt.z);
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
      ctx.lineWidth = 1;
      ctx.moveTo(pt.screenX, pt.screenY);
      ctx.lineTo(floorProj.x, floorProj.y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.fillStyle = 'rgba(100, 116, 139, 0.15)';
      ctx.ellipse(floorProj.x, floorProj.y, 4 * pt.scale, 2 * pt.scale, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // 5. Draw Points and Labels
    for (const pt of sortedPoints) {
      const isHovered = pt.index === state.hoveredIndex;
      const baseRadius = 6 * pt.scale;
      const radius = isHovered ? baseRadius * 1.4 : baseRadius;

      // Color scheme based on category
      const colorFill = THEME_COLORS[pt.category] || THEME_COLORS.feline;

      if (isHovered) {
        ctx.beginPath();
        ctx.arc(pt.screenX, pt.screenY, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(29, 78, 216, 0.22)';
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(pt.screenX, pt.screenY, Math.max(3, radius), 0, Math.PI * 2);
      ctx.fillStyle = colorFill;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      drawPointLabel(pt, isHovered);
    }
  }

  function drawAxisLine(start, end, color, label) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();

    ctx.font = '600 10px var(--font-sans, sans-serif)';
    ctx.fillStyle = color;
    ctx.fillText(label, end.x + 4, end.y + 3);
  }

  function drawPointLabel(pt, isHovered) {
    const labelText = `#${pt.index}: ${truncate(pt.sentence, 26)}`;
    ctx.font = isHovered ? 'bold 11px var(--font-sans, sans-serif)' : '500 10px var(--font-sans, sans-serif)';
    const textWidth = ctx.measureText(labelText).width;
    const padding = 5;
    const pillHeight = 18;
    const pillWidth = textWidth + padding * 2;

    const labelX = pt.screenX + 8;
    const labelY = pt.screenY - 10;

    ctx.fillStyle = isHovered ? 'rgba(15, 23, 42, 0.96)' : 'rgba(255, 255, 255, 0.92)';
    ctx.strokeStyle = isHovered ? '#1d4ed8' : '#cbd5e1';
    ctx.lineWidth = 1;
    roundRect(ctx, labelX, labelY - 13, pillWidth, pillHeight, 4, true, true);

    ctx.fillStyle = isHovered ? '#ffffff' : '#0f172a';
    ctx.fillText(labelText, labelX + padding, labelY);
  }

  function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }

  // =========================================================================
  // INTERACTIVE CONTROLS
  // =========================================================================

  function setupCanvasInteractions() {
    const canvas = elements.canvas;

    canvas.addEventListener('mousedown', (e) => {
      state.camera.isDragging = true;
      state.camera.lastMouseX = e.clientX;
      state.camera.lastMouseY = e.clientY;
    });

    window.addEventListener('mousemove', (e) => {
      if (state.camera.isDragging) {
        const dx = e.clientX - state.camera.lastMouseX;
        const dy = e.clientY - state.camera.lastMouseY;
        state.camera.rotY += dx * 0.008;
        state.camera.rotX += dy * 0.008;
        state.camera.rotX = Math.max(-1.4, Math.min(1.4, state.camera.rotX));
        state.camera.lastMouseX = e.clientX;
        state.camera.lastMouseY = e.clientY;
        renderCanvas3D();
      } else {
        handleCanvasHover(e);
      }
    });

    window.addEventListener('mouseup', () => {
      state.camera.isDragging = false;
    });

    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        state.camera.isDragging = true;
        state.camera.lastMouseX = e.touches[0].clientX;
        state.camera.lastMouseY = e.touches[0].clientY;
      }
    }, { passive: true });

    canvas.addEventListener('touchmove', (e) => {
      if (state.camera.isDragging && e.touches.length === 1) {
        const dx = e.touches[0].clientX - state.camera.lastMouseX;
        const dy = e.touches[0].clientY - state.camera.lastMouseY;
        state.camera.rotY += dx * 0.008;
        state.camera.rotX += dy * 0.008;
        state.camera.rotX = Math.max(-1.4, Math.min(1.4, state.camera.rotX));
        state.camera.lastMouseX = e.touches[0].clientX;
        state.camera.lastMouseY = e.touches[0].clientY;
        renderCanvas3D();
      }
    }, { passive: true });

    canvas.addEventListener('touchend', () => {
      state.camera.isDragging = false;
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomDelta = e.deltaY * -0.0015;
      adjustZoom(zoomDelta);
    }, { passive: false });

    elements.btnZoomIn.addEventListener('click', () => adjustZoom(0.15));
    elements.btnZoomOut.addEventListener('click', () => adjustZoom(-0.15));

    elements.btnResetView.addEventListener('click', () => {
      if (typeof gsap !== 'undefined' && !prefersReducedMotion) {
        gsap.to(state.camera, {
          rotX: -0.32,
          rotY: 0.65,
          zoom: 1.0,
          duration: 0.7,
          ease: 'power2.out',
          onUpdate: renderCanvas3D
        });
      } else {
        state.camera.rotX = -0.32;
        state.camera.rotY = 0.65;
        state.camera.zoom = 1.0;
        renderCanvas3D();
      }
    });
  }

  function adjustZoom(delta) {
    const targetZoom = Math.max(0.5, Math.min(2.5, state.camera.zoom + delta));
    if (typeof gsap !== 'undefined' && !prefersReducedMotion) {
      gsap.to(state.camera, {
        zoom: targetZoom,
        duration: 0.35,
        ease: 'power1.out',
        onUpdate: renderCanvas3D
      });
    } else {
      state.camera.zoom = targetZoom;
      renderCanvas3D();
    }
  }

  function handleCanvasHover(e) {
    const rect = elements.canvas.getBoundingClientRect();
    const scaleX = elements.canvas.width / rect.width;
    const scaleY = elements.canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    let closest = null;
    let minDistance = 16;

    for (const pt of state.points3D) {
      const dist = Math.hypot(pt.screenX - mouseX, pt.screenY - mouseY);
      if (dist < minDistance) {
        minDistance = dist;
        closest = pt;
      }
    }

    if (closest) {
      if (state.hoveredIndex !== closest.index) {
        state.hoveredIndex = closest.index;
        renderCanvas3D();
      }
      showTooltip(closest, e.clientX - rect.left, e.clientY - rect.top);
    } else {
      if (state.hoveredIndex !== -1) {
        state.hoveredIndex = -1;
        renderCanvas3D();
      }
      hideTooltip();
    }
  }

  function showTooltip(pt, clientX, clientY) {
    const tt = elements.pointTooltip;
    tt.innerHTML = `
      <strong>Sentence #${pt.index}</strong><br>
      <em>“${escapeHtml(pt.sentence)}”</em><br>
      <small style="color: #94a3b8;">Coords: [${pt.x.toFixed(1)}, ${pt.y.toFixed(1)}, ${pt.z.toFixed(1)}]</small>
    `;
    tt.style.left = `${clientX}px`;
    tt.style.top = `${clientY}px`;
    tt.classList.remove('hidden');
  }

  function hideTooltip() {
    elements.pointTooltip.classList.add('hidden');
  }

  function updateFlowStep(activeStepIndex, isCompleted) {
    elements.flowSteps.forEach((stepEl, idx) => {
      const stepNum = idx + 1;
      const numBadge = stepEl.querySelector('.step-num');
      if (stepNum < activeStepIndex) {
        stepEl.classList.remove('active');
        stepEl.classList.add('completed');
      } else if (stepNum === activeStepIndex) {
        const wasActive = stepEl.classList.contains('active');
        stepEl.classList.add('active');
        if (isCompleted) stepEl.classList.add('completed');

        if (!wasActive && typeof gsap !== 'undefined' && !prefersReducedMotion && numBadge) {
          gsap.fromTo(
            numBadge,
            { scale: 0.8 },
            { scale: 1.25, duration: 0.25, yoyo: true, repeat: 1, ease: 'power2.out' }
          );
        }
      } else {
        stepEl.classList.remove('active', 'completed');
      }
    });
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function truncate(str, maxLen) {
    return str.length > maxLen ? str.substring(0, maxLen - 1) + '…' : str;
  }

  function init() {
    if (typeof gsap !== 'undefined') {
      const mm = gsap.matchMedia();
      mm.add(
        {
          reduceMotion: '(prefers-reduced-motion: reduce)',
          noPreference: '(prefers-reduced-motion: no-preference)'
        },
        (context) => {
          prefersReducedMotion = !!context.conditions.reduceMotion;
        }
      );
      gsap.defaults({ ease: 'power2.out', duration: 0.4 });
    }

    elements.btnBuildVocab.addEventListener('click', buildVocabulary);
    elements.vocabSearch.addEventListener('input', renderVocabularyTable);

    elements.corpusInput.addEventListener('input', () => {
      state.isVocabBuilt = false;
      elements.vocabStatus.textContent = 'Corpus edited — click "Build Vocabulary" to refresh';
      elements.vocabStatus.className = 'status-indicator';
      invalidateTokenizationResults();
    });

    elements.btnTokenize.addEventListener('click', tokenizeSentence);

    elements.sentenceInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        tokenizeSentence();
      }
    });

    elements.modeSpelling.addEventListener('change', () => {
      state.mode = 'spelling';
      calculateEmbeddings();
    });

    elements.modeMeaning.addEventListener('change', () => {
      state.mode = 'meaning';
      calculateEmbeddings();
    });

    elements.btnPlot3D.addEventListener('click', calculateEmbeddings);

    setupCanvasInteractions();

    buildVocabulary();
    tokenizeSentence();
    calculateEmbeddings();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
