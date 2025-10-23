---
theme: default
background: https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1920
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## BERT Embeddings for Text Classification
  Academic presentation on replacing n-grams with contextual embeddings
drawings:
  persist: false
transition: slide-left
title: BERT Embeddings for Text Classification
mdc: true
---

<div class="flex flex-col items-center justify-center h-full">
  <h1 class="text-5xl font-bold mb-4">Replacing Static N-grams with</h1>
  <h1 class="text-6xl font-bold text-blue-400 mb-12">BERT Embeddings</h1>
  <h2 class="text-3xl mb-8">for Text Classification</h2>
  
  <div class="mt-4 text-xl">
    <p class="mb-2 font-semibold">Group Members</p>
    <p>Feliciann Elliot</p>
    <p>Daryl Nelson</p>
  </div>
  
  <div class="mt-1 text-lg opacity-80">
    <p class="font-semibold">University of Guyana</p>
    <p>MAI5201 - Natural Language Processing</p>
  </div>
</div>

---
layout: default
---

# <span class="text-blue-400">1.</span> Project Overview

<div class="mt-6 text-xl space-y-6 overflow-y-auto max-h-[72vh] pr-2">

**Goal:** Investigate how switching from traditional text features to modern contextual embeddings affects classification performance

<div class="mt-6">

### Three-Way Comparison

<div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-6">
  <div class="p-4 bg-gray-800 rounded-lg card text-white">
    <p class="font-bold text-center">Raw Text</p>
    <p class="text-sm text-center mt-2 opacity-80">Traditional n-grams</p>
  </div>
  <div class="p-4 bg-blue-900 rounded-lg card text-white">
    <p class="font-bold text-center">BERT</p>
    <p class="text-sm text-center mt-2 opacity-80">Full contextual embeddings</p>
  </div>
  <div class="p-4 bg-green-900 rounded-lg card text-white">
    <p class="font-bold text-center">SBERT</p>
    <p class="text-sm text-center mt-2 opacity-80">Optimized embeddings</p>
  </div>
</div>

</div>

<div class="mt-6 p-4 bg-blue-950 rounded-lg card text-white">
  <p class="text-lg"><strong>Evaluation metrics:</strong> Accuracy, Speed, Memory, Overall Performance</p>
</div>

</div>

---
layout: two-cols
---

# <span class="text-blue-400">2.</span> Traditional Text Classification

<div class="mt-6 text-lg space-y-4 overflow-y-auto max-h-[72vh] pr-2">

### How It Works

- Models analyze **n-grams** (word sequences)
- Example: "I love pizza"
  - Bigrams: `["I love", "love pizza"]`
- Counts frequency of occurrence
- Builds statistical patterns

<div class="mt-4 p-2 bg-red-900 rounded-lg card text-white">
  <p class="font-bold text-xl mb-2">⚠️ The Problem</p>
  <p>No semantic understanding</p>
  <p class="text-base mt-2 opacity-90">"love pizza" ≠ "adore pizza"</p>
  <p class="text-base opacity-90">Treats synonyms as completely different</p>
</div>

</div>

::right::

<div class="ml-8 mt-6 overflow-y-auto max-h-[72vh] pr-2">

```python
# Traditional approach
"I love pizza"
↓
["I", "love", "pizza"]
["I love", "love pizza"]
↓
Token counts
{
  "I love": 1,
  "love pizza": 1
}
```

<div class="mt-4 p-2 bg-gray-800 rounded-lg text-base card text-white">
  <p class="font-bold mb-2">Characteristics</p>
  <p>✓ Fast and simple</p>
  <p>✓ Low memory usage</p>
  <p>✗ No context awareness</p>
  <p>✗ Struggles with paraphrasing</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">3.</span> How BERT Changes the Game

<div class="mt-6 text-xl space-y-6 overflow-y-auto max-h-[72vh] pr-2">

<div class="grid grid-cols-1 md:grid-cols-2 gap-3">

<div>

### Context-Aware Understanding

- **Bidirectional encoding** – reads from both directions
- Captures meaning, not just word frequency
- Understands word relationships in context

<div class="mt-4 p-4 bg-blue-950 rounded-lg text-base card text-white">
  
**Example: Context Disambiguation**

```
"river bank" → 🏞️ [0.45, -0.23, 0.78, ...]
"money bank" → 🏦 [0.12, 0.67, -0.34, ...]
```

Same word, different vectors based on context!

</div>

</div>

<div>

### Sentence Representation

Each sentence becomes a **768-dimensional vector** that encodes semantic meaning

```
"I love this restaurant"
        ↓
[0.32, -0.11, 0.75, 0.42, 
 -0.23, 0.58, ..., 0.19]
```

<div class="mt-4 p-4 bg-green-950 rounded-lg text-base card text-white">
  <p class="font-bold mb-2">Key Advantage</p>
  <p>Similar meanings = Similar vectors</p>
  <p class="mt-2 opacity-90">"I love pizza" ≈ "I adore pizza"</p>
</div>

</div>

</div>

</div>

---
layout: default
---

# <span class="text-blue-400">4.</span> Research Questions

<div class="mt-4 space-y-4 overflow-y-auto max-h-[72vh] pr-2">

<div class="p-1 bg-gradient-to-r from-blue-900 to-blue-950 rounded-lg text-xl card text-white">
  <p class="font-bold mb-2">Q1: Performance Enhancement</p>
  <p class="opacity-90">Can simple models like FastText perform better with BERT embeddings instead of word counts?</p>
</div>

<div class="p-1 bg-gradient-to-r from-purple-900 to-purple-950 rounded-lg text-xl card text-white">
  <p class="font-bold mb-2">Q2: Accuracy Gains</p>
  <p class="opacity-90">How much does classification accuracy improve?</p>
</div>

<div class="p-1 bg-gradient-to-r from-green-900 to-green-950 rounded-lg text-xl card text-white">
  <p class="font-bold mb-2">Q3: Computational Cost</p>
  <p class="opacity-90">Does it slow down processing too much for practical use?</p>
</div>

<div class="mt-2 p-1 bg-red-950 rounded-lg text-2xl text-center card text-white">
  <p class="font-bold">Core Question: Is the trade-off worth it?</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">5.</span> Datasets Used

<div class="mt-3 text-lg overflow-y-auto max-h-[72vh] pr-2">

<p class="text-xl mb-6">Real public datasets from Kaggle for comprehensive evaluation</p>

<div class="grid grid-cols-1 md:grid-cols-2 gap-3">

<div class="p-2 bg-blue-950 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2">📰 AG News</p>
  <p class="text-base opacity-90">120,000 news headlines</p>
  <p class="text-sm mt-2 opacity-70">Multi-class news categorization</p>
</div>

<div class="p-2 bg-yellow-950 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2">⭐ Yelp Reviews</p>
  <p class="text-base opacity-90">560,000 sentiment reviews</p>
  <p class="text-sm mt-2 opacity-70">Restaurant review classification</p>
</div>

<div class="p-2 bg-purple-950 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2">❓ Yahoo Answers</p>
  <p class="text-base opacity-90">1.4 million Q&A pairs</p>
  <p class="text-sm mt-2 opacity-70">Question topic classification</p>
</div>

<div class="p-2 bg-orange-950 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2">📦 Amazon Reviews</p>
  <p class="text-base opacity-90">400,000 product reviews</p>
  <p class="text-sm mt-2 opacity-70">Product sentiment analysis</p>
</div>

</div>

<div class="mt-3 p-1 bg-gray-800 rounded-lg text-center card text-white">
  <p class="text-lg"><strong>Common structure:</strong> Text + Category Label</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">6.</span> Pipeline Architecture

<div class="mt-6 text-lg overflow-y-auto max-h-[72vh] pr-2">

<div class="space-y-3">

<div class="flex items-center gap-3">
  <div class="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center font-bold text-lg">1</div>
  <div class="flex-1 p-2 bg-gray-800 rounded-lg card text-white">
    <p class="font-bold">Data Acquisition</p>
    <p class="text-base opacity-80">Automatic dataset download and loading</p>
  </div>
</div>

<div class="flex items-center gap-3">
  <div class="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center font-bold text-lg">2</div>
  <div class="flex-1 p-1 bg-gray-800 rounded-lg card text-white">
    <p class="font-bold">Preprocessing</p>
    <p class="text-base opacity-80">Text cleaning + automatic label column detection</p>
  </div>
</div>

<div class="flex items-center gap-3">
  <div class="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center font-bold text-lg">3</div>
  <div class="flex-1 p-1 bg-gray-800 rounded-lg card text-white">
    <p class="font-bold">Feature Generation (3 versions)</p>
    <p class="text-base opacity-80">Raw text (n-grams) • BERT embeddings • SBERT embeddings</p>
  </div>
</div>

<div class="flex items-center gap-3">
  <div class="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center font-bold text-lg">4</div>
  <div class="flex-1 p-1 bg-gray-800 rounded-lg card text-white">
    <p class="font-bold">Embedding Tokenization</p>
    <p class="text-base opacity-80">Convert vectors to pseudo-tokens: dim1_0.32 dim2_-0.41 ...</p>
  </div>
</div>

<div class="flex items-center gap-3">
  <div class="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center font-bold text-lg">5</div>
  <div class="flex-1 p-3 bg-gray-800 rounded-lg card text-white">
    <p class="font-bold">Model Training & Evaluation</p>
    <p class="text-base opacity-80">FastText training + metrics logging (time, accuracy, precision, recall, F1)</p>
  </div>
</div>

</div>

</div>

---
layout: default
---

# <span class="text-blue-400">7.</span> Baseline: N-gram Approach

<div class="mt-6 text-lg space-y-4 overflow-y-auto max-h-[72vh] pr-2">

<div class="grid grid-cols-1 md:grid-cols-2 gap-3">

<div>

### How It Works

FastText learns patterns based on word and phrase frequency distributions

<div class="mt-4 space-y-1">
  <div class="p-1 bg-green-950 rounded flex items-center gap-3 card text-white">
    <span class="text-2xl">⚡</span>
    <div>
      <p class="font-bold">Speed</p>
      <p class="text-base opacity-80">~25 seconds per dataset</p>
    </div>
  </div>
  
  <div class="p-1 bg-blue-950 rounded flex items-center gap-3 card text-white">
    <span class="text-2xl">🎯</span>
    <div>
      <p class="font-bold">Accuracy</p>
      <p class="text-base opacity-80">~84% on AG News</p>
    </div>
  </div>
  
  <div class="p-1 bg-gray-800 rounded flex items-center gap-3 card text-white">
    <span class="text-2xl">💾</span>
    <div>
      <p class="font-bold">Memory</p>
      <p class="text-base opacity-80">Under 150MB RAM</p>
    </div>
  </div>
</div>

</div>

<div>

### Performance Analysis

<div class="mt-4 p-1 bg-green-900 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2">✓ Strengths</p>
  <ul class="space-y-1 text-base">
    <li>Extremely fast training</li>
    <li>Low computational requirements</li>
    <li>Simple implementation</li>
    <li>Great for prototyping</li>
  </ul>
</div>

<div class="mt-3 p-1 bg-red-900 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2">✗ Weaknesses</p>
  <ul class="space-y-1 text-base">
    <li>No semantic understanding</li>
    <li>Struggles with synonyms</li>
    <li>Poor on paraphrased text</li>
    <li>Limited context awareness</li>
  </ul>
</div>

</div>

</div>

</div>

---
layout: default
---

# <span class="text-blue-400">8.</span> BERT Embedding Approach

<div class="mt-6 text-md space-y-2 overflow-y-auto max-h-[72vh] pr-2">

### The Process

<div class="p-1 bg-blue-950 rounded-lg mb-4 card text-white">
  <p class="text-base">Every sentence passes through BERT → Transformed into a <strong>768-dimensional vector</strong> encoding semantic meaning</p>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-1">

<div class="space-y-1">

**Performance Metrics**

<div class="p-1 h-22 bg-green-950 rounded card text-white">
  <p class="font-bold text-md mb-1">🎯 Accuracy: 91-93%</p>
  <p class="text-base opacity-80">7-9% improvement over n-grams</p>
</div>

<div class="p-1 h-22 bg-orange-950 rounded card text-white">
  <p class="font-bold text-md mb-1">⏱️ Embedding Time</p>
  <p class="text-base opacity-80">10-15 minutes per dataset (CPU)</p>
</div>

<div class="p-1 h-22 bg-purple-950 rounded card text-white">
  <p class="font-bold text-md mb-1">💾 Memory: ~1.8GB</p>
  <p class="text-base opacity-80">Storage for embeddings</p>
</div>

</div>

<div>

**Key Insight**

<div class="p-1 bg-gradient-to-br from-blue-900 to-purple-900 rounded-lg flex flex-col justify-center card text-white">
  <p class="text-xl font-bold mb-2">BERT "understood" language</p>
  <p class="text-md opacity-90 mb-2">FastText had clearer, more meaningful features to learn from</p>
  <p class="text-base opacity-80">→ Significantly reduced classification confusion</p>
  <p class="text-base opacity-80">→ Better generalization to unseen examples</p>
</div>

</div>

</div>

</div>

---
layout: default
---

# <span class="text-blue-400">9.</span> SBERT: The Optimized Solution

<div class="mt-6 text-lg space-y-2 overflow-y-auto max-h-[72vh] pr-2">

<div class="p-2 bg-gradient-to-r from-green-900 to-blue-900 rounded-lg mb-4 card text-white">
  <p class="text-xl font-bold mb-1">Sentence-BERT: Specialized for sentence-level tasks</p>
  <p class="opacity-90">Smaller, faster version optimized for efficient sentence encoding</p>
</div>

<div class="grid grid-cols-1 md:grid-cols-3 gap-3">

<div class="p-2 h-45 bg-green-950 rounded-lg text-center card text-white">
  <p class="text-3xl mb-2">⚡</p>
  <p class="font-bold text-xl mb-1">Speed</p>
  <p class="text-base opacity-80">3-4 minutes</p>
  <p class="text-sm opacity-70 mt-1">~70% faster than BERT</p>
</div>

<div class="p-2 h-45 bg-blue-950 rounded-lg text-center card text-white">
  <p class="text-3xl mb-2">🎯</p>
  <p class="font-bold text-xl mb-1">Accuracy</p>
  <p class="text-base opacity-80">90-92%</p>
  <p class="text-sm opacity-70 mt-1">Nearly identical to BERT</p>
</div>

<div class="p-2 h-45 bg-purple-950 rounded-lg text-center card text-white">
  <p class="text-3xl mb-2">💾</p>
  <p class="font-bold text-xl mb-1">Memory</p>
  <p class="text-base opacity-80">~700MB</p>
  <p class="text-sm opacity-70 mt-1">60% less than BERT</p>
</div>

</div>

<div class="mt-4 p-1 h-45 bg-gradient-to-r from-yellow-900 to-orange-900 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1">🏆 The Sweet Spot</p>
  <p class="text-lg">Same semantic power as BERT at a fraction of the computational cost</p>
  <p class="text-base mt-2 opacity-90"><strong>Ideal for production environments</strong> where both quality and efficiency matter</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">10.</span> Embedding Representation

<div class="mt-6 text-lg overflow-y-auto max-h-[72vh] pr-2">

### From Vectors to Tokens

<div class="mt-3 p-2 bg-gray-800 rounded-lg card text-white text-sm leading-snug space-y-2">

  <div><strong>Original sentence:</strong></div>
  <pre class="whitespace-pre-wrap text-sm leading-snug">"This restaurant has amazing food"</pre>

  <div><strong>BERT embedding (768 dimensions):</strong></div>
  <pre class="whitespace-pre-wrap text-sm leading-snug">[0.32, -0.11, 0.75, 0.42, -0.23, ..., 0.19]</pre>

  <div><strong>Tokenized representation for FastText:</strong></div>
  <pre class="whitespace-pre-wrap text-sm leading-snug">dim0_0.32 dim1_-0.11 dim2_0.75 dim3_0.42 dim4_-0.23 ... dim767_0.19</pre>

</div>

<div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">

<div class="p-2 bg-blue-950 rounded-lg card text-white">
  <p class="font-bold text-xl mb-2">Why This Works</p>
  <p class="text-base opacity-90">FastText treats each embedding dimension as a learnable "token"</p>
  <p class="text-base opacity-90 mt-2">Combines deep contextual understanding with shallow classifier efficiency</p>
</div>

<div class="p-2 bg-purple-950 rounded-lg card text-white">
  <p class="font-bold text-xl mb-2">The Bridge</p>
  <p class="text-base opacity-90">This tokenization bridges two worlds:</p>
  <p class="text-base mt-2">🧠 <strong>Deep models</strong> (BERT's understanding)</p>
  <p class="text-base">⚡ <strong>Shallow classifiers</strong> (FastText's speed)</p>
</div>

</div>

</div>

---
layout: default
---

# <span class="text-blue-400">11.</span> Evaluation Metrics

<div class="mt-6 text-md overflow-y-auto max-h-[72vh] pr-2">

<div class="grid grid-cols-1 md:grid-cols-2 gap-3">

<div class="space-y-2">

<div class="p-2 bg-blue-950 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1 h-4">🎯 Accuracy</p>
  <p class="opacity-90">Percentage of correct predictions</p>
  <p class="text-base mt-1 opacity-70 italic">How often is the model right?</p>
</div>

<div class="p-2 bg-green-950 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1 h-4">✓ Precision</p>
  <p class="opacity-90">Quality of positive predictions</p>
  <p class="text-base mt-1 opacity-70 italic">When it predicts positive, is it correct?</p>
</div>

<div class="p-2 bg-purple-950 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1 h-4">🔍 Recall</p>
  <p class="opacity-90">Coverage of actual positives</p>
  <p class="text-base mt-1 opacity-70 italic">Does it find all the positive cases?</p>
</div>

</div>

<div class="space-y-3">

<div class="p-2 bg-orange-950 rounded-lg card text-white">
  <p class="text-xl font-bold mb-1">⚖️ F1-Score</p>
  <p class="opacity-90">Harmonic mean of precision & recall</p>
  <p class="text-base mt-1 opacity-70 italic">Overall balance metric</p>
</div>

<div class="p-2 bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg flex flex-col justify-center card text-white">
  <p class="font-bold text-xl mb-1 h-4">📊 Output</p>
  <p class="opacity-90">All metrics logged and saved to CSV for easy comparison across approaches</p>
  <p class="text-base mt-1 opacity-70">Enables rigorous statistical analysis</p>
</div>

</div>

</div>

</div>

---
layout: default
---

# <span class="text-blue-400">12.</span> Experimental Results

<div class="mt-6 overflow-y-auto max-h-[72vh] pr-2">

<div class="overflow-auto">

| Model Type | Avg Accuracy | Train Time | Embedding Time | Notes |
|------------|--------------|------------|----------------|-------|
| **FastText (n-grams)** | 84% | 25s | 0s | Fastest, weakest |
| **FastText + BERT** | 92% | 40s | 12m | Strongest but slow |
| **FastText + SBERT** | 91% | 35s | 4m | Best speed-accuracy mix |

</div>

<div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">

<div class="p-4 bg-gray-800 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2 text-center">N-grams</p>
  <div class="text-center text-5xl mb-2">⚡</div>
  <p class="text-center opacity-80">Speed champion</p>
  <p class="text-center text-sm mt-1 opacity-60">but limited understanding</p>
</div>

<div class="p-4 bg-blue-900 rounded-lg card text-white">
  <p class="text-xl font-bold mb-2 text-center">BERT</p>
  <div class="text-center text-5xl mb-2">🎯</div>
  <p class="text-center opacity-80">Accuracy champion</p>
  <p class="text-center text-sm mt-1 opacity-60">at the cost of time</p>
</div>

<div class="p-4 bg-green-900 rounded-lg border-4 border-yellow-500 card text-white">
  <p class="text-xl font-bold mb-2 text-center">SBERT</p>
  <div class="text-center text-5xl mb-2">🏆</div>
  <p class="text-center opacity-80">Overall winner</p>
  <p class="text-center text-sm mt-1 opacity-60">optimal balance</p>
</div>

</div>

<div class="mt-4 p-4 bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg text-center card text-white">
  <p class="text-2xl font-bold">Context helps significantly, but at the price of embedding time</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">13.</span> Efficiency Analysis

<div class="mt-6 text-lg overflow-y-auto max-h-[72vh] pr-2">

### Throughput Comparison (CPU)

<div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">

<div class="p-4 bg-green-950 rounded-lg text-center card text-white">
  <p class="font-bold text-2xl mb-2">N-grams</p>
  <p class="text-5xl font-bold mb-2">50K</p>
  <p class="text-xl opacity-80">samples/minute</p>
  <div class="mt-2 text-base opacity-70">
    <p>✓ Fastest processing</p>
    <p>✓ No preprocessing overhead</p>
  </div>
</div>

<div class="p-4 bg-red-950 rounded-lg text-center card text-white">
  <p class="font-bold text-2xl mb-2">BERT</p>
  <p class="text-5xl font-bold mb-2">6K</p>
  <p class="text-xl opacity-80">samples/minute</p>
  <div class="mt-2 text-base opacity-70">
    <p>✗ Slowest processing</p>
    <p>✗ High compute requirements</p>
  </div>
</div>

<div class="p-4 bg-blue-950 rounded-lg text-center border-2 border-blue-400 card text-white">
  <p class="font-bold text-2xl mb-2">SBERT</p>
  <p class="text-5xl font-bold mb-2">18K</p>
  <p class="text-xl opacity-80">samples/minute</p>
  <div class="mt-2 text-base opacity-70">
    <p>✓ 3x faster than BERT</p>
    <p>✓ Manageable overhead</p>
  </div>
</div>

</div>

<div class="mt-4 p-4 bg-gradient-to-r from-purple-900 to-pink-900 rounded-lg card text-white">
  <p class="text-xl font-bold mb-1">⚡ GPU Acceleration</p>
  <p class="opacity-90">On GPU, these speeds triple — making contextual embeddings much more practical for production use</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">14.</span> Resource Requirements

<div class="mt-6 text-lg overflow-y-auto max-h-[72vh] pr-2">

<div class="grid grid-cols-1 md:grid-cols-2 gap-3">

<div>

### Memory Footprint

<div class="space-y-3 mt-3">

<div class="p-3 bg-green-950 rounded-lg flex justify-between items-center card text-white">
  <div>
    <p class="font-bold text-xl">N-gram Model</p>
    <p class="text-base opacity-80">Minimal footprint</p>
  </div>
  <p class="text-3xl font-bold">150MB</p>
</div>

<div class="p-3 bg-red-950 rounded-lg flex justify-between items-center card text-white">
  <div>
    <p class="font-bold text-xl">BERT Embeddings</p>
    <p class="text-base opacity-80">Highest requirement</p>
  </div>
  <p class="text-3xl font-bold">1.8GB</p>
</div>

<div class="p-3 bg-blue-950 rounded-lg flex justify-between items-center border-2 border-blue-400 card text-white">
  <div>
    <p class="font-bold text-xl">SBERT Embeddings</p>
    <p class="text-base opacity-80">Much more manageable</p>
  </div>
  <p class="text-3xl font-bold">700MB</p>
</div>

</div>

</div>

<div>

### Hardware Implications

<div class="mt-3 space-y-3">

<div class="p-2 bg-gray-800 rounded-lg card text-white">
  <p class="font-bold text-xl mb-1">💻 Limited Hardware</p>
  <p class="opacity-90">SBERT is the clear winner for resource-constrained environments</p>
</div>

<div class="p-2 bg-blue-950 rounded-lg card text-white">
  <p class="font-bold text-xl mb-1">🖥️ Production Systems</p>
  <p class="opacity-90">SBERT offers best balance of memory efficiency and performance</p>
</div>

<div class="p-2 bg-purple-950 rounded-lg card text-white">
  <p class="font-bold text-xl mb-1 h-3">🚀 High-Performance</p>
  <p class="opacity-90">BERT viable with sufficient GPU memory and compute</p>
</div>

</div>

</div>

</div>

</div>

---
layout: default
---

# <span class="text-blue-400">15.</span> Key Observations

<div class="mt-6 text-lg space-y-3 overflow-y-auto max-h-[72vh] pr-2">

<div class="p-2 bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1 h-3">1️⃣ Contextual Embeddings Enhance Simple Models</p>
  <p class="opacity-90">Even a basic FastText classifier becomes significantly smarter with BERT/SBERT features</p>
</div>

<div class="p-2 bg-gradient-to-r from-green-900 to-teal-900 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1 h-3">2️⃣ BERT Achieves Highest Accuracy</p>
  <p class="opacity-90">Full BERT embeddings provide the best classification performance (~92%)</p>
</div>

<div class="p-2 bg-gradient-to-r from-yellow-900 to-orange-900 rounded-lg border-4 border-yellow-500 card text-white">
  <p class="text-2xl font-bold mb-1 h-3">3️⃣ SBERT Hits the Sweet Spot</p>
  <p class="opacity-90">Optimal balance between speed and quality — practical for real-world deployment</p>
</div>

<div class="p-2 bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1 h-3">4️⃣ N-grams: Limited but Fast</p>
  <p class="opacity-90">Good for quick prototypes, but inadequate for nuanced language understanding</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">16.</span> Practical Implications

<div class="mt-6 text-lg space-y-4 overflow-y-auto max-h-[72vh] pr-2">

<div class="p-4 bg-gradient-to-br from-purple-900 to-blue-900 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1 h-5">💡 The Core Concept</p>
  <p class="text-xl opacity-90">Using embeddings = giving FastText "super-charged features"</p>
  <p class="mt-2 opacity-80">Instead of raw words → we feed it semantic meanings</p>
</div>

<div class="mt-4">
  <p class="text-2xl font-bold mb-2">Real-World Applications</p>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
    <div class="p-3 bg-blue-950 rounded-lg card text-white h-25">
      <p class="font-bold text-xl mb-1">📧 Customer Support</p>
      <p class="text-base opacity-80">Accurate ticket classification and routing</p>
    </div>
    <div class="p-3 bg-green-950 rounded-lg card text-white h-25">
      <p class="font-bold text-xl mb-1">⭐ Review Analysis</p>
      <p class="text-base opacity-80">Better sentiment detection and categorization</p>
    </div>
    <div class="p-3 bg-purple-950 rounded-lg card text-white h-25">
      <p class="font-bold text-xl mb-1">🚫 Spam Detection</p>
      <p class="text-base opacity-80">Understands context, not just keywords</p>
    </div>
    <div class="p-3 bg-orange-950 rounded-lg card text-white h-25">
      <p class="font-bold text-xl mb-1">📰 Content Moderation</p>
      <p class="text-base opacity-80">Catches nuanced policy violations</p>
    </div>
  </div>
</div>

<div class="mt-4 p-4 bg-red-950 rounded-lg text-center card text-white">
  <p class="text-xl font-bold">Result: Fewer misclassifications & more reliable predictions</p>
</div>

</div>

---
layout: default
---

# <span class="text-blue-400">17.</span> Conclusion & Future Work

<div class="mt-6 text-lg space-y-4 overflow-y-auto max-h-[72vh] pr-2">

<div class="p-4 bg-gradient-to-r from-green-900 to-blue-900 rounded-lg card text-white">
  <p class="text-2xl font-bold mb-1">🎯 Main Finding</p>
  <p class="text-xl opacity-90">Replacing static n-grams with BERT embeddings improves accuracy by <strong>7-10% overall</strong></p>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-3">

<div class="p-2 bg-orange-950 rounded-lg card text-white">
  <p class="font-bold text-xl mb-1">⚖️ Trade-offs</p>
  <ul class="space-y-1 text-base opacity-90">
    <li>✓ Significant accuracy gains</li>
    <li>✗ Longer embedding time</li>
    <li>✗ Higher memory requirements</li>
    <li>✓ More robust to paraphrasing</li>
  </ul>
</div>

<div class="p-2 bg-green-950 rounded-lg border-2 border-green-400 card text-white">
  <p class="font-bold text-xl mb-1">🏆 Recommendation</p>
  <p class="text-base opacity-90">Use <strong>SBERT</strong> for production systems:</p>
  <ul class="space-y-1 text-base opacity-90 mt-1">
    <li>• ~91% accuracy (nearly BERT-level)</li>
    <li>• 3x faster than full BERT</li>
    <li>• Manageable resource footprint</li>
  </ul>
</div>

</div>

</div>

---
layout: center
class: text-center
---

# Thank You

<div class="mt-6 text-2xl space-y-4">

<p class="text-3xl font-bold text-blue-400">Questions?</p>

<div class="mt-6 text-lg opacity-80">
  <p class="font-semibold">Feliciann Elliot • Daryl Nelson</p>
  <p class="mt-4">University of Guyana</p>
  <p>MAI5201 - Natural Language Processing</p>
</div>

</div>

<style>
/* Gradient headline style from your original */
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}

/* Make slides scroll instead of clipping if content is tall */
.slidev-layout { overflow: auto; }

/* GLOBAL BOX TEXT -> FORCE WHITE (element + descendants) */
.slidev-layout :is(.card,[class*="bg-"],[class*="from-"],[class*="to-"]) { color:#fff !important; }
.slidev-layout :is(.card,[class*="bg-"],[class*="from-"],[class*="to-"]) * { color:#fff !important; }
.slidev-layout .text-black, .slidev-layout .text-gray-900, .slidev-layout .text-slate-900 { color:#fff !important; }

/* Shrink padding inside *every* card/box */
:root { --box-p: 0.5rem; } /* ~p-2 */
.card { padding: var(--box-p) !important; border-radius: 0.5rem !important; color:#fff !important; }
.card * { color: inherit !important; }

/* Also shrink default padding utilities when combined with card */
.card.p-4, .card.p-5, .card.p-6, .card.p-3 { padding: var(--box-p) !important; }
.card.p-8, .card.p-10 { padding: var(--box-p) !important; }

/* Cap grid boxes so they don't balloon */
.grid > .card { max-height: 26vh; overflow:auto; }
.grid { align-items: start; }

/* Prevent code blocks from overflowing horizontally */
pre, code { max-width: 100%; overflow: auto; }
</style>
