# XAI Attention Visualizer

**AIPI 590.01 - Explainable AI Final Project**

This project analyzes attention patterns in transformer models to detect hallucinations in real-time. When LLMs generate unreliable text, their attention shows specific signatures: diffuse focus (high entropy), unstable drift across layers, and weak token grounding. It also compares this pattern across different models. 

For now, we've used this on the DistilGPT2 and the GPT2 models, since they are relatively smaller and faster to run and demonstrate. 

## Live Demo

**[View Live Application](https://huggingface.co/spaces/Jog-sama/xai-attention-visualize)**

## What This Does

Ever wonder how LLMs confidently spew out complete nonsense? This tool visualizes what's happening under the hood. When a model generates text, it pays attention to different tokens with varying intensities. Hallucinated text has a different attention signature than well-grounded responses.

**Important Note:** This tool uses small models (DistilGPT2, GPT2) which generate low-quality text frequently. The attention-based analysis framework demonstrates the methodology and provides proof-of-concept for hallucination detection. With larger, better-calibrated models, the patterns would be clearer and more reliable.

### Two Modes:

1. **Input Analysis** - Analyze how the model processes any text you provide (shows attention distribution, useful for exploring patterns)
2. **Generation Analysis** - Model generates a response and we track attention token-by-token (demonstrates real-time hallucination risk indicators)

### Features:

- Attention heatmap video showing how focus evolves across layers
- Token-by-token entropy tracking during generation
- Color-coded token importance (green = well-grounded, red = weak connections)
- Attention drift measurement (stability across layers)
- Multi-factor hallucination risk assessment
- Comprehensive analysis reports

## Running Locally

### Prerequisites

You need:
- Python 3.11+
- FFmpeg (for video generation)

### Install FFmpeg

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

### Setup

```bash
# clone the repo
git clone https://github.com/ag826/Understanding_Attention_XAI_F25.git
cd Understanding_Attention_XAI_F25

# install dependencies
pip install -r requirements.txt

# run it
python app.py
```

Open http://localhost:5000 in your browser.

## Deploying to Render (Free)

This app is configured for easy deployment on Render.

1. Push code to GitHub
2. Go to https://render.com and sign up
3. Click "New +" and select "Web Service"
4. Connect your GitHub repo
5. Render will auto-detect the configuration
6. Click "Create Web Service"

That's it. Render will install FFmpeg automatically using `apt-packages.txt`.

**Note:** The free tier spins down after inactivity. First request after idle might take 30-60 seconds to wake up.

**Live Demo also available on:** https://huggingface.co/spaces/Jog-sama/xai-attention-visualize

## How It Works

### The Core Idea

When transformer models generate text, they use attention mechanisms to decide which input tokens are relevant for predicting the next token. We analyze these attention patterns to detect hallucinations.

**Key Signals:**

1. **Entropy** - Measures how focused vs scattered the attention is. High entropy means the model is attending to many tokens with similar weights (diffuse, uncertain). Low entropy means focused attention (confident).

2. **Drift** - Measures how much attention patterns change across layers. High drift suggests unstable reasoning pathways.

3. **Token Importance** - How much attention each token receives. Tokens with very low importance are weakly grounded in the input.

**The Insight:** When models hallucinate, they may show characteristic attention patterns - high entropy (scattered attention suggesting uncertainty) or extremely low token importance (weak grounding). However, small models generate unreliable text frequently, making these patterns noisy. This framework provides a methodology for attention-based analysis that could be more effective with larger models.

### Implementation

The tool uses distilGPT2 and GPT2 (small models, fast inference). Here's what happens:

**Generation Mode:**
1. You provide a prompt
2. Model generates a response token-by-token
3. We capture attention weights at each generation step
4. Calculate entropy for each generated token
5. Spikes in entropy = potential hallucination points

**LIME-Inspired Token Importance:**
We sum the attention each token receives across all layers and positions, then normalize. This gives us a score for how "grounded" each token is.

**Multi-Factor Risk:**
- Entropy > 2.5 = high risk
- Drift > 0.08 = high risk  
- Problematic token ratio > 25% = high risk
- 2+ factors triggered = HIGH RISK overall

### Why Small Models?

DistilGPT2 and GPT2 are small enough to run on free tier hosting, making this tool accessible for demonstrations. However, they generate low-quality text and hallucinate frequently, which makes them imperfect for evaluating detection accuracy. They serve as proof-of-concept models to demonstrate the methodology.

The attention analysis framework we've built (token-by-token tracking, multi-factor risk assessment, entropy/drift measurements) applies to any transformer model. Larger models like GPT-3.5 or GPT-4 would show clearer patterns, as they're better calibrated and hallucinate less frequently.

## Project Structure

```
.
├── app.py                  # Flask backend, main logic, original version
├── app_gradio.py           # Gradio specific version just for huggingface deployment
├── templates/
│   └── index.html         # Frontend UI
├── requirements.txt       # Python dependencies
├── Procfile              # Deployment config
├── runtime.txt           # Python version
├── apt-packages.txt      # System packages (FFmpeg)
└── README.md            # This file
```

## Example Prompts

Try these in **Generate Response** mode to see the tool in action:

**Simple prompts:**
- "Where does the sun set?"
- "How many legs does a spider have?"
- "What color is the sky?"

**Nonsense:**
- "humans have 5 hands"
- "the moon is made of cheese"

The model will generate responses (often low-quality or incorrect), and you can observe the attention patterns, entropy metrics, and risk indicators. Note that small models generate unreliable text frequently, so the tool demonstrates the analytical framework rather than providing definitive hallucination detection.

## Limitations

- **Small models have noisy signals:** DistilGPT2 and GPT2 generate unreliable text frequently, making it difficult to distinguish clear hallucination patterns. They often show low entropy even when generating incorrect information (false confidence).
- **Input mode limitations:** Analyzing pre-written text doesn't detect hallucinations - it only shows how the model processes that text. Longer sentences naturally show higher entropy/drift regardless of accuracy.
- **Not ground-truth detection:** This tool analyzes attention patterns that may correlate with hallucinations, but cannot definitively determine if generated text is factually correct.
- **Video generation latency:** Creating attention heatmap videos takes 1-2 minutes for longer sequences
- **Free tier constraints:** Limited memory on free hosting restricts model size

**What this tool IS:** A framework for attention-based analysis with token-by-token tracking, demonstrating a methodology for hallucination risk assessment.

**What this tool ISN'T:** A production-ready hallucination detector or fact-checker.

## Technical Notes

**Token-by-Token Tracking:**
During generation, `model.generate()` with `output_attentions=True` returns attention weights for each step. We extract the last layer's attention pattern for each newly generated token and calculate its entropy.

**Relative Thresholds:**
Instead of absolute thresholds (which flag 97% of tokens as problematic), we use percentile-based thresholds. Bottom 30% = problematic, top 40% = high importance.

**Video Generation:**
FFmpeg creates MP4 files showing attention heatmaps evolving across layers. Each frame = one layer, 1 FPS for easy viewing.

## Course Alignment

This project directly applies concepts from AIPI 590.01:

- Attention mechanism analysis (core XAI technique)
- LIME-inspired feature importance
- Entropy as an uncertainty measure
- Real-time model behavior explanation
- Interactive visualization for interpretability

**Novel Component:** Token-by-token attention tracking during generation. Most XAI tools analyze static inputs. We track attention dynamically as the model generates each token, providing a framework for real-time analysis. While small models produce noisy signals, this methodology demonstrates an approach that could be more effective with larger, better-calibrated models.