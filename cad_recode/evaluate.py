# cad_recode/evaluate.py
import argparse
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import cadquery as cq
from cad_recode.dataset import CadRecodeDataset
from cad_recode.model   import CADRecodeModel
from cad_recode.utils   import sample_points_on_shape, chamfer_distance, save_point_cloud, edit_distance

# --------------------------------------------------------------------------- #
# Monte-Carlo IoU fallback
# --------------------------------------------------------------------------- #
def approximate_iou_via_sampling(solid_a, solid_b, n_samples=25000):
    """
    Estimate IoU by Monte-Carlo sampling inside the union’s axis-aligned bounding box.
    """
    if solid_a is None or solid_b is None:
        return 0.0
    bb_a = solid_a.BoundingBox()
    bb_b = solid_b.BoundingBox()
    xmin = min(bb_a.xmin, bb_b.xmin)
    xmax = max(bb_a.xmax, bb_b.xmax)
    ymin = min(bb_a.ymin, bb_b.ymin)
    ymax = max(bb_a.ymax, bb_b.ymax)
    zmin = min(bb_a.zmin, bb_b.zmin)
    zmax = max(bb_a.zmax, bb_b.zmax)
    if xmax - xmin < 1e-9 or ymax - ymin < 1e-9 or zmax - zmin < 1e-9:
        return 0.0
    pts = np.random.rand(n_samples, 3)
    pts[:, 0] = pts[:, 0] * (xmax - xmin) + xmin
    pts[:, 1] = pts[:, 1] * (ymax - ymin) + ymin
    pts[:, 2] = pts[:, 2] * (zmax - zmin) + zmin
    def inside(shape, p):
        return shape.distToShape(cq.Vector(*p))[0] < 1e-9
    inside_a = np.fromiter((inside(solid_a, p) for p in pts), dtype=bool, count=n_samples)
    inside_b = np.fromiter((inside(solid_b, p) for p in pts), dtype=bool, count=n_samples)
    inter = np.logical_and(inside_a, inside_b).sum()
    union = np.logical_or(inside_a, inside_b).sum()
    return inter / union if union > 0 else 0.0

# --------------------------------------------------------------------------- #
# Evaluation routine
# --------------------------------------------------------------------------- #
def evaluate_model(model, dataset, *,
                   batch_size       = 1,
                   max_length       = 256,
                   num_candidates   = 1,
                   device           = None,
                   save_examples    = 0):
    """
    Evaluate `model` on `dataset`. Returns a metrics dict and list of example predictions.
    Metrics include Chamfer distance, IoU, token accuracy, and edit distance.
    If save_examples > 0, collects that many example predictions (code and point clouds).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    chamfer_scores = []
    iou_scores = []
    token_accuracies = []
    edit_distances = []
    invalid = 0
    examples = []
    saved_count = 0

    for pts_batch, code_true_batch in loader:
        torch.cuda.empty_cache()
        pts_batch = pts_batch.to(device)   # (B, 256, 3)
        B = pts_batch.size(0)

        # Generate candidate codes
        try:
            init_emb, attn_mask = model.prepare_prefix(pts_batch)
            gen = model.decoder.generate(
                inputs_embeds        = init_emb,
                attention_mask       = attn_mask,
                max_length           = max_length,
                num_beams            = num_candidates,
                num_return_sequences = num_candidates,
                early_stopping       = True,
                eos_token_id         = model.end_id,
            )
        except Exception as e:
            print(f"[WARN] Generation failed for batch of size {B}: {e}")
            invalid += B
            continue

        # gen shape = (B * num_candidates, T)
        gen = gen.view(B, num_candidates, -1)

        for b in range(B):
            cand_codes = []
            for j in range(num_candidates):
                seq = gen[b, j].clone()
                # Remove leading <|start|> if present
                if seq[0].item() == model.start_id:
                    seq = seq[1:]
                # Trim at end-token
                end_idxs = (seq == model.end_id).nonzero(as_tuple=True)[0]
                if len(end_idxs):
                    seq = seq[:end_idxs[0]+1]
                code_str = model.tokenizer.decode(seq, skip_special_tokens=False)
                cand_codes.append(code_str)

            # Find best candidate by Chamfer distance
            best_cd = float("inf")
            best_shape = None
            best_code = None
            for code_pred in cand_codes:
                try:
                    loc = {}
                    exec(code_pred, {"cq": cq}, loc)
                    shp = loc.get("result") or loc.get("r") or loc.get("shape")
                    if isinstance(shp, cq.Workplane):
                        shp = shp.val()
                except Exception:
                    continue
                if shp is None:
                    continue
                pts_pred = sample_points_on_shape(shp, 1024)
                pts_true = pts_batch[b].cpu().numpy()  # (256,3)
                cd = chamfer_distance(pts_pred, pts_true)
                if cd < best_cd:
                    best_cd = cd
                    best_shape = shp
                    best_code = code_pred

            if best_shape is None or best_code is None:
                invalid += 1
                continue

            # Record Chamfer distance
            chamfer_scores.append(best_cd)

            # Compute IoU (exact or fallback)
            try:
                solid_pred = best_shape.val() if isinstance(best_shape, cq.Workplane) else best_shape
                loc_gt = {}
                exec(code_true_batch[b], {"cq": cq}, loc_gt)
                solid_gt = loc_gt.get("result") or loc_gt.get("r") or loc_gt.get("shape")
                if isinstance(solid_gt, cq.Workplane):
                    solid_gt = solid_gt.val()
                vol_pred = solid_pred.Volume()
                vol_gt   = solid_gt.Volume()
                inter    = solid_pred.intersect(solid_gt)
                vol_int  = inter.Volume() if inter else 0.0
                vol_union = vol_pred + vol_gt - vol_int
                iou = vol_int / vol_union if vol_union > 1e-9 else 0.0
            except Exception:
                iou = approximate_iou_via_sampling(solid_pred, solid_gt)
            iou_scores.append(iou)

            # Compute token accuracy
            # Clean code strings (remove special tokens if any)
            code_pred_clean = best_code.replace("<|start|>", "").replace("<|end|>", "").strip()
            code_true_str = code_true_batch[b]
            # Tokenize to IDs
            true_ids = model.tokenizer.encode(code_true_str, add_special_tokens=False)
            pred_ids = model.tokenizer.encode(code_pred_clean, add_special_tokens=False)
            # Compute per-token accuracy
            matches = sum(int(t == p) for t, p in zip(true_ids, pred_ids))
            total_tokens = len(true_ids)
            tok_acc = matches / total_tokens if total_tokens > 0 else 0.0
            token_accuracies.append(tok_acc)
            # Compute edit distance (token-level)
            dist = edit_distance(true_ids, pred_ids)
            edit_distances.append(dist)

            # Save example predictions if requested
            if save_examples and saved_count < save_examples:
                try:
                    loc_gt = {}
                    exec(code_true_str, {"cq": cq}, loc_gt)
                    solid_gt = loc_gt.get("result") or loc_gt.get("r") or loc_gt.get("shape")
                    if isinstance(solid_gt, cq.Workplane):
                        solid_gt = solid_gt.val()
                    pts_true_full = sample_points_on_shape(solid_gt, 1024)
                    pts_pred_full = sample_points_on_shape(best_shape, 1024)
                except Exception:
                    continue
                examples.append({
                    "code_true": code_true_str,
                    "code_pred": code_pred_clean,
                    "pts_true": pts_true_full,
                    "pts_pred": pts_pred_full
                })
                saved_count += 1

    # Aggregate metrics
    metrics = {
        "mean_chamfer": float(np.mean(chamfer_scores)) if chamfer_scores else None,
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else None,
        "invalid_ratio": float(invalid / len(dataset)),
        "mean_token_accuracy": float(np.mean(token_accuracies)) if token_accuracies else None,
        "mean_edit_distance": float(np.mean(edit_distances)) if edit_distances else None
    }

    print(f"[Eval] Chamfer Distance (mean): {metrics['mean_chamfer']}")
    print(f"[Eval] IoU             (mean): {metrics['mean_iou']}")
    print(f"[Eval] Token Accuracy   (mean): {metrics['mean_token_accuracy']}")
    print(f"[Eval] Edit Distance    (mean): {metrics['mean_edit_distance']}")
    print(f"[Eval] Invalid samples: {invalid}/{len(dataset)} ({metrics['invalid_ratio']*100:.2f}%)")

    return metrics, examples

# --------------------------------------------------------------------------- #
# CLI wrapper
# --------------------------------------------------------------------------- #
def main(args):
    # Load dataset
    test_set = CadRecodeDataset(args.data_root, split=args.split,
                                n_points=256, noise_std=0.0, noise_prob=0.0)
    # Load model
    model = CADRecodeModel(llm_name=args.llm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        # our training script saved {'model': state_dict, 'optim': …, …}
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state_dict)
    # Evaluate
    metrics, examples = evaluate_model(
        model,
        dataset=test_set,
        batch_size=1,
        max_length=args.max_length,
        num_candidates=args.num_candidates,
        device=device,
        save_examples=args.save_examples
    )
    # Determine output directories
    run_dir = args.output_dir or os.path.dirname(args.checkpoint) or "."
    os.makedirs(run_dir, exist_ok=True)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    # Save metrics to JSON
    results_file = os.path.join(run_dir, "eval_results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {results_file}")

    # Save example predictions
    for idx, ex in enumerate(examples):
        base = os.path.join(eval_dir, f"example_{idx:03d}")
        # Save true and predicted code
        true_code_file = base + "_true.py"
        pred_code_file = base + "_pred.py"
        with open(true_code_file, 'w') as f:
            f.write(ex["code_true"] + "\n")
        with open(pred_code_file, 'w') as f:
            f.write(ex["code_pred"] + "\n")
        # Save point clouds as PLY
        ply_true = base + "_true.ply"
        ply_pred = base + "_pred.ply"
        save_point_cloud(ex["pts_true"], ply_true)
        save_point_cloud(ex["pts_pred"], ply_pred)
    if examples:
        print(f"Saved {len(examples)} example predictions to {eval_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CAD-Recode model")
    parser.add_argument("--data_root",   required=True, help="Path to dataset directory")
    parser.add_argument("--checkpoint",  required=True, help="Path to trained model .pt")
    parser.add_argument("--split",       default="val", choices=["train","val","test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--llm",         default="Qwen/Qwen2-1.5B", help="LLM model name")
    parser.add_argument("--max_length",  type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--num_candidates", type=int, default=1,
                        help="Number of candidate outputs (beam search)")
    parser.add_argument("--output_dir",  default=None,
                        help="Directory to save evaluation results (default=checkpoint directory)")
    parser.add_argument("--save_examples", type=int, default=5,
                        help="Number of example predictions (code + PLY) to save")
    args = parser.parse_args()
    main(args)
