"""
XAI Attention Visualizer - Gradio Version
AIPI 590.01 Final Project

Gradio version for Hugging Face Spaces deployment (always-on, free)
Note: This version was specifically adapted for Gradio (Hugging Face Spaces) deployment
For the original version, refer to app.py
Citation: Claude was used for the generation of this gradio specific code adaptation from the original app.py
"""

import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.animation import FuncAnimation, FFMpegWriter
import uuid
import tempfile
import os

# model cache
mdl_cache = {}

def load_mdl(mdl_name):
    """loads and caches model"""
    if mdl_name not in mdl_cache:
        print(f"loading model: {mdl_name}")
        tok = AutoTokenizer.from_pretrained(mdl_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            mdl_name,
            output_attentions=True,
            output_hidden_states=False
        )
        mdl_cache[mdl_name] = (tok, mdl)
    return mdl_cache[mdl_name]

def analyze_text(input_text, model_choice, analysis_mode):
    """main analysis function"""
    
    # model name mapping
    model_map = {
        "DistilGPT2 (Fast)": "distilgpt2",
        "GPT2 (Standard)": "gpt2"
    }
    mdl_name = model_map[model_choice]
    
    # load model
    tok, mdl = load_mdl(mdl_name)
    
    if analysis_mode == "Analyze Input Text":
        # input mode - just analyze the text
        inputs = tok(input_text, return_tensors="pt")
        tokens = tok.convert_ids_to_tokens(inputs['input_ids'][0])
        
        with torch.no_grad():
            outputs = mdl(**inputs, output_attentions=True)
        
        attn_weights = outputs.attentions
        generated_text = None
        gen_steps = []
        
    else:
        # generate mode
        inputs = tok(input_text, return_tensors="pt")
        
        with torch.no_grad():
            gen_outputs = mdl.generate(
                **inputs,
                max_new_tokens=30,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tok.eos_token_id
            )
        
        generated_ids = gen_outputs.sequences[0]
        generated_text = tok.decode(generated_ids, skip_special_tokens=True)
        tokens = tok.convert_ids_to_tokens(generated_ids)
        
        # track token-by-token
        gen_steps = []
        if gen_outputs.attentions:
            prompt_len = len(tok.convert_ids_to_tokens(inputs['input_ids'][0]))
            for step_i, step_attn in enumerate(gen_outputs.attentions):
                last_layer = step_attn[-1][0]
                avg_attn = last_layer.mean(dim=0).detach().cpu().numpy()
                new_tok_attn = avg_attn[-1, :]
                attn_probs = new_tok_attn / (new_tok_attn.sum() + 1e-10)
                ent = -np.sum(attn_probs * np.log(attn_probs + 1e-10))
                
                gen_tok = tokens[prompt_len + step_i] if prompt_len + step_i < len(tokens) else "?"
                gen_tok = gen_tok.replace('Ġ', ' ')
                
                level = "HIGH" if ent > 3.0 else "MEDIUM" if ent > 2.0 else "LOW"
                gen_steps.append(f"{gen_tok} | Entropy: {ent:.3f} | {level}")
        
        # get full attention
        full_inputs = tok(generated_text, return_tensors="pt")
        with torch.no_grad():
            full_outputs = mdl(**full_inputs, output_attentions=True)
        attn_weights = full_outputs.attentions
    
    # process tokens
    display_tokens = [t.replace('Ġ', ' ') for t in tokens]
    
    # calculate metrics
    avg_matrices = []
    attn_stats = []
    
    for idx, layer_tensor in enumerate(attn_weights):
        if layer_tensor is None:
            continue
        avg_matrix = layer_tensor[0].mean(dim=0).detach().cpu().numpy()
        avg_matrices.append(avg_matrix)
        
        entropy = -np.sum(avg_matrix * np.log(avg_matrix + 1e-10), axis=1).mean()
        attn_stats.append({
            'layer': idx,
            'entropy': float(entropy)
        })
    
    # calculate drift
    drifts = []
    layer_mats = [m.flatten() for m in avg_matrices]
    for i in range(len(layer_mats) - 1):
        diff = np.abs(layer_mats[i+1] - layer_mats[i]).mean()
        drifts.append(float(diff))
    avg_drift = np.mean(drifts) if drifts else 0.0
    
    # token importance
    tok_importance = []
    for layer_tensor in attn_weights:
        if layer_tensor is None:
            continue
        avg_matrix = layer_tensor[0].mean(dim=0).detach().cpu().numpy()
        importance = avg_matrix.sum(axis=0)
        tok_importance.append(importance)
    
    avg_importance = np.mean(tok_importance, axis=0)
    if avg_importance.max() > 0:
        norm_importance = avg_importance / avg_importance.max()
    else:
        norm_importance = avg_importance
    
    # relative thresholds
    sorted_imp = np.sort(norm_importance)
    low_thresh = np.percentile(sorted_imp, 30)
    
    prob_count = sum(1 for imp in norm_importance if imp < low_thresh)
    prob_ratio = prob_count / len(norm_importance) if len(norm_importance) > 0 else 0
    
    # risk assessment
    avg_ent = np.mean([s['entropy'] for s in attn_stats])
    ent_risk = avg_ent > 2.5
    drift_risk = avg_drift > 0.08
    tok_risk = prob_ratio > 0.25
    
    risk_score = sum([ent_risk, drift_risk, tok_risk])
    
    if prob_ratio > 0.4:
        risk_level = "High"
    elif risk_score >= 2:
        risk_level = "High"
    elif risk_score == 1:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    # create video
    fig, ax = plt.subplots(figsize=(10, 8))
    im = sns.heatmap(
        avg_matrices[0],
        ax=ax,
        xticklabels=display_tokens,
        yticklabels=display_tokens,
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Attention Weight'}
    )
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    hmap_coll = im.collections[0]
    
    def update_frame(layer_num):
        matrix = avg_matrices[layer_num]
        hmap_coll.set_array(matrix.flatten())
        ax.set_title(f'Layer {layer_num + 1}/{len(avg_matrices)}')
        return [hmap_coll]
    
    ani = FuncAnimation(fig, update_frame, frames=len(avg_matrices), interval=1000)
    
    # save video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    writer = FFMpegWriter(fps=1.0)
    ani.save(temp_file.name, writer=writer)
    plt.close(fig)
    
    # format output
    stats_text = f"""
## Attention Statistics
- **Layers:** {len(avg_matrices)}
- **Tokens:** {len(display_tokens)}
- **Avg Entropy:** {avg_ent:.2f} ({'high' if ent_risk else 'normal'})
- **Avg Drift:** {avg_drift:.3f} ({'high' if drift_risk else 'normal'})
- **Problematic Tokens:** {prob_ratio:.1%}

## Risk Assessment
**Risk Level: {risk_level}**

{'Critical: High proportion of weak tokens' if prob_ratio > 0.4 else 
 'Multiple hallucination indicators detected' if risk_score >= 2 else
 'Some indicators present' if risk_score == 1 else
 'Patterns appear stable'}
"""
    
    # token display with colors
    token_html = "<div style='padding: 10px; background: #2a2a2a; border-radius: 8px;'>"
    token_html += "<h3>Token Importance</h3>"
    for i, (tok, imp) in enumerate(zip(display_tokens, norm_importance)):
        color = '#8bc34a' if imp >= 0.6 else '#ffb74d' if imp >= 0.3 else '#ff6b6b'
        token_html += f"<span style='background: {color}; padding: 5px 10px; margin: 3px; border-radius: 4px; display: inline-block;'>{tok}</span>"
    token_html += "</div>"
    
    # generation steps
    gen_text = ""
    if generated_text:
        gen_text = f"\n## Generated Response\n**Prompt:** {input_text}\n**Response:** {generated_text}\n\n"
        if gen_steps:
            gen_text += "### Token-by-Token Entropy\n"
            gen_text += "\n".join(gen_steps[:20])  # limit to first 20
    
    output_text = gen_text + stats_text
    
    return temp_file.name, output_text, token_html

# create gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🔍 XAI Attention Visualizer")
    gr.Markdown("Analyzing attention patterns in transformer models to detect hallucination indicators")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter text or question",
                placeholder="humans have 5 hands",
                lines=3
            )
            model_choice = gr.Radio(
                choices=["DistilGPT2 (Fast)", "GPT2 (Standard)"],
                value="DistilGPT2 (Fast)",
                label="Select Model"
            )
            analysis_mode = gr.Radio(
                choices=["Analyze Input Text", "Generate Response & Analyze"],
                value="Analyze Input Text",
                label="Analysis Mode"
            )
            analyze_btn = gr.Button("Analyze Attention Patterns", variant="primary")
    
    with gr.Row():
        video_output = gr.Video(label="Attention Evolution Video")
    
    with gr.Row():
        with gr.Column():
            stats_output = gr.Markdown(label="Analysis Results")
        with gr.Column():
            token_output = gr.HTML(label="Token Importance")
    
    gr.Markdown("""
    ### How to Use
    1. Enter text (or a question for generation mode)
    2. Choose model and analysis mode
    3. Click "Analyze"
    
    **Note:** Small models generate unreliable text frequently. This tool demonstrates the methodology 
    for attention-based analysis. Patterns would be clearer with larger models.
    """)
    
    analyze_btn.click(
        fn=analyze_text,
        inputs=[input_text, model_choice, analysis_mode],
        outputs=[video_output, stats_output, token_output]
    )

if __name__ == "__main__":
    demo.launch()