#!/usr/bin/env python3
"""
Evaluation script to test QuickDraw (qd) scripts and store results.
Tests functionality and runs evaluations on the qd scripts.
"""

import os
import json
import time
import random
import traceback
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import clip
from PIL import Image, ImageDraw


# ============================================================================
# Test Configuration
# ============================================================================

TEST_CONFIG = {
    "samples_per_class": 10,  # Small number for quick testing
    "classes": ["cat", "car", "apple", "tree", "house"],
    "test_classes": ["cat", "car"],  # Classes to test with
    "output_dir": "outputs",
    "results_file": None,  # Will be set with timestamp
}


# ============================================================================
# Shared Utilities
# ============================================================================

def render_strokes(drawing, size=256, stroke_width=3, padding=10):
    """Render QuickDraw strokes to an image."""
    xs = np.concatenate([np.array(s[0]) for s in drawing])
    ys = np.concatenate([np.array(s[1]) for s in drawing])
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w, h = max_x - min_x + 1e-6, max_y - min_y + 1e-6
    scale = (size - 2 * padding) / max(w, h)
    ox = (size - scale * w) / 2
    oy = (size - scale * h) / 2

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for s in drawing:
        x, y = np.array(s[0]), np.array(s[1])
        X, Y = (x - min_x) * scale + ox, (y - min_y) * scale + oy
        pts = list(zip(X, Y))
        if len(pts) >= 2:
            draw.line(pts, fill="black", width=stroke_width, joint="curve")
    return img


# ============================================================================
# Test Functions
# ============================================================================

def test_imports() -> Dict[str, Any]:
    """Test that all required imports work."""
    print("Testing imports...")
    result = {
        "test": "imports",
        "status": "passed",
        "errors": []
    }
    
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import clip  # noqa: F401
        from PIL import Image, ImageDraw  # noqa: F401
        print("  ✓ All imports successful")
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        print(f"  ✗ Import error: {e}")
    
    return result


def test_data_loading() -> Dict[str, Any]:
    """Test that QuickDraw data files can be loaded."""
    print("\nTesting data loading...")
    result = {
        "test": "data_loading",
        "status": "passed",
        "errors": [],
        "files_found": [],
        "files_missing": []
    }
    
    data_dir = Path("data/quickdraw/strokes")
    if not data_dir.exists():
        result["status"] = "failed"
        result["errors"].append(f"Data directory not found: {data_dir}")
        print(f"  ✗ Data directory not found: {data_dir}")
        return result
    
    for class_name in TEST_CONFIG["test_classes"]:
        file_path = data_dir / f"{class_name}.ndjson"
        if file_path.exists():
            try:
                with open(file_path) as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        result["files_found"].append({
                            "class": class_name,
                            "samples": len(lines)
                        })
                        print(f"  ✓ Found {class_name}.ndjson ({len(lines)} samples)")
                    else:
                        result["files_missing"].append(class_name)
                        print(f"  ✗ {class_name}.ndjson is empty")
            except Exception as e:
                result["status"] = "failed"
                result["errors"].append(f"Error reading {class_name}: {e}")
                print(f"  ✗ Error reading {class_name}: {e}")
        else:
            result["files_missing"].append(class_name)
            print(f"  ✗ {class_name}.ndjson not found")
    
    if result["files_missing"]:
        result["status"] = "partial"
    
    return result


def test_rendering() -> Dict[str, Any]:
    """Test that rendering function works."""
    print("\nTesting rendering...")
    result = {
        "test": "rendering",
        "status": "passed",
        "errors": []
    }
    
    try:
        # Create a simple test drawing
        test_drawing = [[[0, 10, 20], [0, 10, 20]]]
        img = render_strokes(test_drawing)
        
        assert img.size == (256, 256), "Image size incorrect"
        assert img.mode == "RGB", "Image mode incorrect"
        print("  ✓ Rendering successful")
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        print(f"  ✗ Rendering error: {e}")
        traceback.print_exc()
    
    return result


def test_clip_loading() -> Dict[str, Any]:
    """Test that CLIP model can be loaded."""
    print("\nTesting CLIP model loading...")
    result = {
        "test": "clip_loading",
        "status": "passed",
        "errors": [],
        "device": None,
        "model_name": "ViT-B/32"
    }
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
        
        result["device"] = device
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        
        print(f"  ✓ CLIP loaded on {device}")
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        print(f"  ✗ CLIP loading error: {e}")
        traceback.print_exc()
    
    return result


def test_single_prediction() -> Dict[str, Any]:
    """Test a single prediction (like qd_zeroshot_one.py)."""
    print("\nTesting single prediction...")
    result = {
        "test": "single_prediction",
        "status": "passed",
        "errors": [],
        "true_label": None,
        "predicted_label": None,
        "correct": None,
        "similarities": {}
    }
    
    try:
        # Load CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        
        # Load a random sample
        classes = TEST_CONFIG["test_classes"]
        label = random.choice(classes)
        result["true_label"] = label
        
        path = f"data/quickdraw/strokes/{label}.ndjson"
        with open(path) as f:
            lines = f.readlines()
        sample = json.loads(random.choice(lines))
        
        img = render_strokes(sample["drawing"]).convert("RGB")
        
        # Get embeddings
        templates = [
            "{}", "a sketch of a {}", "a line drawing of a {}",
            "a simple doodle of a {}", "an outline of a {}"
        ]
        
        with torch.no_grad():
            # Image embedding
            x = preprocess(img).unsqueeze(0).to(device)
            img_emb = model.encode_image(x)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            
            # Text embeddings
            per_class = []
            for c in classes:
                toks = clip.tokenize([t.format(c) for t in templates]).to(device)
                e = model.encode_text(toks)
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(dim=0, keepdim=True)
                e = e / e.norm(dim=-1, keepdim=True)
                per_class.append(e)
            text_embs = torch.vstack(per_class)
            
            # Predict
            sims = (img_emb @ text_embs.T).squeeze(0)
            pred_idx = sims.argmax().item()
            result["predicted_label"] = classes[pred_idx]
            result["correct"] = (pred_idx == classes.index(label))
            
            for c, s in zip(classes, sims.tolist()):
                result["similarities"][c] = float(s)
        
        print(f"  ✓ Prediction: True={label}, Pred={result['predicted_label']}, Correct={result['correct']}")
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        print(f"  ✗ Prediction error: {e}")
        traceback.print_exc()
    
    return result


def test_batch_evaluation() -> Dict[str, Any]:
    """Test batch evaluation (like qd_zeroshot_eval.py)."""
    print("\nTesting batch evaluation...")
    result = {
        "test": "batch_evaluation",
        "status": "passed",
        "errors": [],
        "overall_accuracy": None,
        "per_class_accuracy": {},
        "total_samples": 0,
        "correct_samples": 0,
        "runtime_seconds": None
    }
    
    try:
        # Load CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        
        classes = TEST_CONFIG["test_classes"]
        K = TEST_CONFIG["samples_per_class"]
        templates = [
            "{}", "a sketch of a {}", "a line drawing of a {}",
            "a simple doodle of a {}", "an outline of a {}"
        ]
        
        # Build text embeddings
        with torch.no_grad():
            per_class_text = []
            for c in classes:
                toks = clip.tokenize([t.format(c) for t in templates]).to(device)
                e = model.encode_text(toks)
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(dim=0, keepdim=True)
                e = e / e.norm(dim=-1, keepdim=True)
                per_class_text.append(e)
            text_embs = torch.vstack(per_class_text)
        
        # Evaluate
        start_time = time.time()
        correct, total = 0, 0
        per_class_correct = {c: 0 for c in classes}
        per_class_total = {c: 0 for c in classes}
        
        for ci, c in enumerate(classes):
            path = f"data/quickdraw/strokes/{c}.ndjson"
            with open(path) as f:
                lines = f.readlines()
            
            n = min(K, len(lines))
            idxs = random.sample(range(len(lines)), k=n)
            
            for i in idxs:
                sample = json.loads(lines[i])
                img = render_strokes(sample["drawing"]).convert("RGB")
                
                with torch.no_grad():
                    x = preprocess(img).unsqueeze(0).to(device)
                    v = model.encode_image(x)
                    v = v / v.norm(dim=-1, keepdim=True)
                    pred = (v @ text_embs.T).argmax(dim=1).item()
                
                is_correct = int(pred == ci)
                correct += is_correct
                total += 1
                per_class_correct[c] += is_correct
                per_class_total[c] += 1
        
        end_time = time.time()
        runtime = end_time - start_time
        
        result["overall_accuracy"] = correct / total if total > 0 else 0.0
        result["correct_samples"] = correct
        result["total_samples"] = total
        result["runtime_seconds"] = runtime
        
        for c in classes:
            class_acc = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0
            result["per_class_accuracy"][c] = {
                "accuracy": class_acc,
                "correct": per_class_correct[c],
                "total": per_class_total[c]
            }
        
        print(f"  ✓ Batch evaluation: Accuracy={result['overall_accuracy']:.4f} ({correct}/{total})")
        print(f"    Runtime: {runtime:.2f} seconds")
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        print(f"  ✗ Batch evaluation error: {e}")
        traceback.print_exc()
    
    return result


def test_prompt_comparison() -> Dict[str, Any]:
    """Test prompt comparison (like qd_random_zeroshot.py)."""
    print("\nTesting prompt comparison...")
    result = {
        "test": "prompt_comparison",
        "status": "passed",
        "errors": [],
        "true_label": None,
        "best_prompt": None,
        "prompt_scores": {}
    }
    
    try:
        # Load CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        
        classes = TEST_CONFIG["test_classes"]
        label = random.choice(classes)
        result["true_label"] = label
        
        # Load sample
        path = f"data/quickdraw/strokes/{label}.ndjson"
        with open(path) as f:
            lines = f.readlines()
        sample = json.loads(random.choice(lines))
        img = render_strokes(sample["drawing"]).convert("RGB")
        
        # Create prompts: plain vs descriptive
        classes_descriptive = [f"sketch of a {c}" for c in classes]
        prompts = [p for pair in zip(classes, classes_descriptive) for p in pair]
        
        with torch.no_grad():
            # Image embedding
            x = preprocess(img).unsqueeze(0).to(device)
            img_emb = model.encode_image(x)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            
            # Text embeddings
            toks = clip.tokenize(prompts).to(device)
            text_embs = model.encode_text(toks)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            
            # Similarities
            sims = (img_emb @ text_embs.T).squeeze(0)
            best_idx = sims.argmax().item()
            result["best_prompt"] = prompts[best_idx]
            
            for prompt, score in zip(prompts, sims.tolist()):
                result["prompt_scores"][prompt] = float(score)
        
        print(f"  ✓ Prompt comparison: Best={result['best_prompt']}")
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        print(f"  ✗ Prompt comparison error: {e}")
        traceback.print_exc()
    
    return result


# ============================================================================
# Main Execution
# ============================================================================

def run_all_tests() -> Dict[str, Any]:
    """Run all tests and collect results."""
    print("=" * 70)
    print("QuickDraw Scripts Evaluation")
    print("=" * 70)
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": TEST_CONFIG,
        "tests": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "partial": 0
        }
    }
    
    # Run tests
    test_functions = [
        test_imports,
        test_data_loading,
        test_rendering,
        test_clip_loading,
        test_single_prediction,
        test_batch_evaluation,
        test_prompt_comparison,
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            all_results["tests"].append(result)
            all_results["summary"]["total_tests"] += 1
            
            if result["status"] == "passed":
                all_results["summary"]["passed"] += 1
            elif result["status"] == "failed":
                all_results["summary"]["failed"] += 1
            elif result["status"] == "partial":
                all_results["summary"]["partial"] += 1
        except Exception as e:
            print(f"\n  ✗ Test {test_func.__name__} crashed: {e}")
            traceback.print_exc()
            all_results["tests"].append({
                "test": test_func.__name__,
                "status": "failed",
                "errors": [str(e)]
            })
            all_results["summary"]["total_tests"] += 1
            all_results["summary"]["failed"] += 1
    
    return all_results


def save_results(results: Dict[str, Any]) -> str:
    """Save results to JSON file."""
    os.makedirs(TEST_CONFIG["output_dir"], exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    filepath = os.path.join(TEST_CONFIG["output_dir"], filename)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    return filepath


def print_summary(results: Dict[str, Any]):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    summary = results["summary"]
    print(f"Total tests: {summary['total_tests']}")
    print(f"  ✓ Passed:  {summary['passed']}")
    print(f"  ✗ Failed:  {summary['failed']}")
    print(f"  ~ Partial:  {summary['partial']}")
    print("=" * 70)


if __name__ == "__main__":
    # Run all tests
    results = run_all_tests()
    
    # Save results
    results_file = save_results(results)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    exit_code = 0 if results["summary"]["failed"] == 0 else 1
    exit(exit_code)

