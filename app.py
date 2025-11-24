"""
XAI Attention Visualizer
AIPI 590.01 Final Project

This project analyzes attention patterns in transformer models to detect hallucinations.
The core observation is that when models generate unreliable text, their attention shows specific patterns:
diffuse focus (high entropy), unstable drift, weak token connections. It also compares this pattern across varios models.
"""

from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import uuid
from datetime import datetime
import json

app = Flask(__name__)

os.makedirs('static/videos', exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# building model cache so we don't reload every time
mdl_cache = {}

class NumpyEncoder(json.JSONEncoder):
    """function since numpy types aren't JSON serializable by default"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def load_mdl(mdl_name):
    """loads model and tokenizer, caches them"""
    if mdl_name not in mdl_cache:
        print(f"loading model: {mdl_name}")
        tok = AutoTokenizer.from_pretrained(mdl_name)
        
        # gotta force output_attentions here otherwise it doesn't work
        mdl = AutoModelForCausalLM.from_pretrained(
            mdl_name,
            output_attentions=True,
            output_hidden_states=False
        )
        
        mdl_cache[mdl_name] = (tok, mdl)
    
    return mdl_cache[mdl_name]

def gen_and_analyze(prompt_text, mdl_name="distilgpt2", max_tokens=30):
    """
    generates response and tracks attention token by token
    this is where hallucination detection actually happens
    """
    tok, mdl = load_mdl(mdl_name)
    
    inputs = tok(prompt_text, return_tensors="pt")
    prompt_toks = tok.convert_ids_to_tokens(inputs['input_ids'][0])
    plen = len(prompt_toks)
    
    print(f"prompt: {prompt_text}")
    print(f"generating...")
    
    # generate with attention tracking
    with torch.no_grad():
        outs = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tok.eos_token_id
        )
    
    gen_ids = outs.sequences[0]
    gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
    all_toks = tok.convert_ids_to_tokens(gen_ids)
    disp_toks = [t.replace('Ġ', ' ') for t in all_toks]
    
    print(f"generated: {gen_txt}")
    print(f"total tokens: {len(disp_toks)}")
    
    # track attention for each token as it's generated
    # this is the key part - we're looking at what the model focuses on
    # when it generates each word
    gen_steps = []
    if outs.attentions and len(outs.attentions) > 0:
        print(f"got attention for {len(outs.attentions)} steps")
        
        for step_i, step_attn in enumerate(outs.attentions):
            # grab last layer (most relevant for final output)
            last_layer = step_attn[-1][0]
            
            # average across heads
            avg_attn = last_layer.mean(dim=0).detach().cpu().numpy()
            
            # the new token's attention pattern
            new_tok_attn = avg_attn[-1, :]
            
            # entropy calculation - higher = more scattered attention = bad sign
            attn_probs = new_tok_attn / (new_tok_attn.sum() + 1e-10)
            ent = -np.sum(attn_probs * np.log(attn_probs + 1e-10))
            
            curr_len = plen + step_i + 1
            gen_tok = disp_toks[plen + step_i] if curr_len <= len(disp_toks) else "?"
            
            gen_steps.append({
                'step': step_i,
                'token': gen_tok,
                'entropy': float(ent),
                'attention_distribution': attn_probs.tolist(),
                'max_attention': float(new_tok_attn.max()),
                'position': plen + step_i
            })
            
            print(f"step {step_i}: token '{gen_tok}', entropy={ent:.3f}")
    
    # now get full attention for complete sequence (for video visualization)
    full_inp = tok(gen_txt, return_tensors="pt")
    with torch.no_grad():
        full_out = mdl(**full_inp, output_attentions=True)
    
    full_attn = full_out.attentions
    
    return {
        'prompt': prompt_text,
        'generated_text': gen_txt,
        'full_text': gen_txt,
        'tokens': disp_toks,
        'attention_weights': full_attn,
        'prompt_length': plen,
        'generated_length': len(disp_toks) - plen,
        'generation_steps': gen_steps
    }

def make_attn_video(txt, mdl_name="distilgpt2", mode="input"):
    """
    main function for generating attention viz
    
    mode = 'input': analyzes text you give it
    mode = 'generate': generates response and analyzes that
    """
    if mode == "generate":
        # generate and track
        gen_res = gen_and_analyze(txt, mdl_name)
        toks = gen_res['tokens']
        disp_toks = toks
        attn_wts = gen_res['attention_weights']
        
        gen_info = {
            'mode': 'generate',
            'prompt': gen_res['prompt'],
            'generated_text': gen_res['generated_text'],
            'prompt_length': gen_res['prompt_length'],
            'generated_length': gen_res['generated_length'],
            'generation_steps': gen_res.get('generation_steps', [])
        }
    else:
        # just analyze input
        tok, mdl = load_mdl(mdl_name)
        
        inps = tok(txt, return_tensors="pt")
        toks = tok.convert_ids_to_tokens(inps['input_ids'][0])
        disp_toks = [t.replace('Ġ', ' ') for t in toks]
        
        with torch.no_grad():
            outs = mdl(**inps, output_attentions=True)
        
        attn_wts = outs.attentions
        gen_info = {'mode': 'input'}
    
    print(f"attn weights type: {type(attn_wts)}")
    print(f"attn weights len: {len(attn_wts) if attn_wts else 0}")
    
    if attn_wts is None or len(attn_wts) == 0:
        raise ValueError("model didn't return attention weights")
    
    n_layers = len(attn_wts)
    print(f"processing {n_layers} layers")
    
    # prep attention matrices
    avg_mats = []
    attn_stats = []
    
    for i, layer_t in enumerate(attn_wts):
        print(f"layer {i} type: {type(layer_t)}, shape: {layer_t.shape if layer_t is not None else 'None'}")
        
        if layer_t is None:
            print(f"layer {i} is None, skipping")
            continue
            
        try:
            # average across heads
            avg_mat = layer_t[0].mean(dim=0).detach().cpu().numpy()
            avg_mats.append(avg_mat)
            
            # calc entropy for this layer
            ent = -np.sum(avg_mat * np.log(avg_mat + 1e-10), axis=1).mean()
            attn_stats.append({
                'layer': i,
                'entropy': float(ent),
                'max_attention': float(avg_mat.max()),
                'mean_attention': float(avg_mat.mean())
            })
        except Exception as e:
            print(f"error processing layer {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # analyze token importance (LIME-ish approach)
    tok_analysis = calc_tok_importance(attn_wts, disp_toks)
    
    # calc drift across layers
    drift_res = calc_drift(attn_wts)
    
    # make the video
    fig, ax = plt.subplots(figsize=(10, 8))
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0.0, vmax=1.0)
    
    im = sns.heatmap(
        avg_mats[0],
        ax=ax,
        xticklabels=disp_toks,
        yticklabels=disp_toks,
        cmap='viridis',
        linewidths=.5,
        linecolor='lightgray',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0.0,
        vmax=1.0,
        norm=norm
    )
    ax.set_xlabel('Key (Attended To) Tokens', fontsize=11)
    ax.set_ylabel('Query (Attending) Tokens', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    hmap_coll = im.collections[0]
    
    def update_frame(layer_n):
        """updates heatmap for each layer"""
        mat = avg_mats[layer_n]
        hmap_coll.set_array(mat.flatten())
        ax.set_title(f'Layer {layer_n + 1}/{n_layers} - Average Attention Weights', 
                     fontsize=13, fontweight='bold')
        return [hmap_coll]
    
    # animate
    ani = FuncAnimation(fig, update_frame, frames=n_layers, interval=1000, blit=False)
    
    # save as mp4
    vid_id = str(uuid.uuid4())
    out_path = f'static/videos/attention_{vid_id}.mp4'
    
    writer = FFMpegWriter(fps=1.0, metadata=dict(artist='XAI'), bitrate=1800)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    
    return {
        'video_path': out_path,
        'tokens': disp_toks,
        'token_analysis': tok_analysis,
        'attention_stats': attn_stats,
        'drift_analysis': drift_res,
        'num_layers': n_layers,
        'model_name': mdl_name,
        'generation_info': gen_info
    }

def calc_tok_importance(attn_wts, toks):
    """
    LIME-inspired token importance
    basically sums up how much attention each token gets
    """
    tok_imp = []
    
    for li, lt in enumerate(attn_wts):
        avg_mat = lt[0].mean(dim=0).detach().cpu().numpy()
        # sum columns (attention received)
        imp = avg_mat.sum(axis=0)
        tok_imp.append(imp)
    
    # average across layers
    avg_imp = np.mean(tok_imp, axis=0)
    
    # normalize
    if avg_imp.max() > 0:
        norm_imp = avg_imp / avg_imp.max()
    else:
        norm_imp = avg_imp
    
    # relative thresholds so we don't get 97% problematic every time
    tok_res = []
    
    sorted_imp = np.sort(norm_imp)
    high_t = np.percentile(sorted_imp, 60)
    low_t = np.percentile(sorted_imp, 30)
    
    for idx, (tok, imp) in enumerate(zip(toks, norm_imp)):
        is_prob = imp < low_t
        
        if imp >= high_t:
            cat = 'high_attention'
        elif imp >= low_t:
            cat = 'medium_attention'
        else:
            cat = 'low_attention'
            
        tok_res.append({
            'token': tok,
            'importance': float(imp),
            'position': idx,
            'is_problematic': is_prob,
            'category': cat
        })
    
    return tok_res

def calc_drift(attn_wts):
    """
    measures how much attention changes across layers
    high drift = unstable reasoning
    """
    layer_mats = []
    for lt in attn_wts:
        avg_mat = lt[0].mean(dim=0).detach().cpu().numpy()
        layer_mats.append(avg_mat.flatten())
    
    # diffs between consecutive layers
    drifts = []
    for i in range(len(layer_mats) - 1):
        diff = np.abs(layer_mats[i+1] - layer_mats[i]).mean()
        drifts.append(float(diff))
    
    avg_drift = np.mean(drifts)
    return {
        'layer_drifts': drifts,
        'average_drift': float(avg_drift),
        'interpretation': (
            "high drift suggests unstable attention patterns, "
            "which might indicate uncertain or hallucinated reasoning"
        )
    }

def analyze_risk(attn_stats, tok_analysis, drift_res):
    """
    combines entropy + drift + token quality to assess hallucination risk
    """
    avg_ent = np.mean([s['entropy'] for s in attn_stats])
    avg_drift = drift_res['average_drift']
    
    # count problematic tokens
    prob_toks = sum(1 for t in tok_analysis if t['is_problematic'])
    tot_toks = len(tok_analysis)
    prob_ratio = prob_toks / tot_toks if tot_toks > 0 else 0
    
    # multi-factor assessment
    # lowered thresholds since even "good" outputs show some entropy/drift
    ent_risk = avg_ent > 2.5  # changed from 3.0
    drift_risk = avg_drift > 0.08  # changed from 0.12
    tok_risk = prob_ratio > 0.25  # changed from 0.35 (if 25%+ tokens weak = risky)
    
    risk_score = sum([ent_risk, drift_risk, tok_risk])
    
    # special case for really bad token quality
    if prob_ratio > 0.4:  # changed from 0.5
        risk_lvl = "High"
        expl = f"critical: {prob_ratio:.1%} of tokens show weak grounding, severe hallucination or incoherent generation"
    elif risk_score >= 2:
        risk_lvl = "High"
        expl = "multiple hallucination indicators detected: "
        factors = []
        if ent_risk:
            factors.append("high attention entropy")
        if drift_risk:
            factors.append("unstable attention drift")
        if tok_risk:
            factors.append("many weak token connections")
        expl += ", ".join(factors)
    elif risk_score == 1:
        risk_lvl = "Medium"
        expl = "some hallucination indicators present, not conclusive though"
    else:
        risk_lvl = "Low"
        expl = "attention patterns appear stable and well-grounded"
    
    res = {
        'average_entropy': float(avg_ent),
        'average_drift': float(avg_drift),
        'problematic_token_ratio': float(prob_ratio),
        'risk_level': risk_lvl,
        'risk_score': risk_score,
        'interpretation': expl,
        'detailed_analysis': {
            'entropy': f"{avg_ent:.2f} ({'high' if ent_risk else 'normal'})",
            'drift': f"{avg_drift:.3f} ({'high' if drift_risk else 'normal'})",
            'token_quality': f"{prob_ratio:.1%} problematic tokens"
        }
    }
    
    return res

def make_summary(input_txt, mdl_name, tok_analysis, attn_stats, drift_res, risk_res):
    """generates summary report"""
    prob_toks = [t for t in tok_analysis if t['is_problematic']]
    high_imp_toks = sorted(tok_analysis, key=lambda x: x['importance'], reverse=True)[:5]
    
    report = {
        'input_text': input_txt,
        'model': mdl_name,
        'summary': {
            'total_tokens': len(tok_analysis),
            'total_layers': len(attn_stats),
            'overall_risk': risk_res['risk_level'],
            'key_metrics': risk_res['detailed_analysis']
        },
        'problematic_tokens': [
            f"{t['token']} (pos {t['position']}, importance {t['importance']:.2f})"
            for t in prob_toks[:5]
        ],
        'most_influential_tokens': [
            f"{t['token']} (importance {t['importance']:.2f})"
            for t in high_imp_toks
        ],
        'layer_insights': {
            'entropy_trend': 'increasing' if attn_stats[-1]['entropy'] > attn_stats[0]['entropy'] else 'decreasing',
            'most_focused_layer': min(enumerate(attn_stats), key=lambda x: x[1]['entropy'])[0],
            'most_diffuse_layer': max(enumerate(attn_stats), key=lambda x: x[1]['entropy'])[0],
        },
        'recommendations': [
            "review low-attention tokens as they might indicate weak grounding" if risk_res['risk_score'] >= 2 else "attention patterns appear stable",
            f"layer {attn_stats.index(min(attn_stats, key=lambda x: x['entropy']))} shows strongest focus" if len(attn_stats) > 0 else "N/A"
        ]
    }
    
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        txt = data.get('text', '')
        mdl = data.get('model', 'distilgpt2')
        mode = data.get('mode', 'input')
        
        print(f"received: text='{txt}', model='{mdl}', mode='{mode}'")
        
        if not txt:
            return jsonify({'error': 'no input text'}), 400
        
        print("loading model...")
        res = make_attn_video(txt, mdl, mode=mode)
        print("video generated")
        
        print("analyzing risk...")
        risk_res = analyze_risk(
            res['attention_stats'],
            res['token_analysis'],
            res['drift_analysis']
        )
        
        print("generating summary...")
        summary = make_summary(
            txt,
            mdl,
            res['token_analysis'],
            res['attention_stats'],
            res['drift_analysis'],
            risk_res
        )
        
        resp = {
            'success': True,
            'video_url': '/' + res['video_path'],
            'tokens': res['tokens'],
            'token_analysis': res['token_analysis'],
            'attention_stats': res['attention_stats'],
            'drift_analysis': res['drift_analysis'],
            'hallucination_analysis': risk_res,
            'summary_report': summary,
            'generation_info': res.get('generation_info', {'mode': 'input'}),
            'timestamp': datetime.now().isoformat()
        }
        
        print("analysis complete")
        return app.response_class(
            response=json.dumps(resp, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        import traceback
        err_det = traceback.format_exc()
        print(f"error: {str(e)}")
        print(f"traceback:\n{err_det}")
        return jsonify({'error': f'{str(e)}'}), 500

@app.route('/models')
def get_models():
    mdls = [
        {'name': 'distilgpt2', 'display': 'DistilGPT2 (Fast)'},
        {'name': 'gpt2', 'display': 'GPT2 (Standard)'},
    ]
    return jsonify(mdls)

if __name__ == '__main__':
    print("=" * 50)
    print("XAI Attention Visualizer")
    print("=" * 50)
    
    # check ffmpeg
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print("✓ FFmpeg installed")
    except FileNotFoundError:
        print("✗ FFmpeg not found - install it first")
    
    print(f"✓ working dir: {os.getcwd()}")
    
    try:
        import torch
        print(f"✓ pytorch {torch.__version__}")
    except ImportError:
        print("✗ pytorch not installed")
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed")
    
    print("=" * 50)
    print("starting server on port 5000")
    print("=" * 50)
    
    app.run(debug=False, host='0.0.0.0', port=5000)

# AI Citation - ChatGPT and Claude were used to generate initial code structure and framework in app.py
# But not more than 5 lines at a time 
# Most of the code and logic was manually written 
# AI Citation - ChatGPT and Claude were used to help debug and optimize code snippets in app.py
# The above code is well documented through comments and function descriptions
# For a more detailed explanation, please refer to the README.md file in the repository
# There, I have explained the project, function and core logic in detail.