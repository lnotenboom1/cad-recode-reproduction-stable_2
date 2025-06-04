# cad_recode/train.py
"""Unified, human‑readable training script for CAD‑Recode on DelftBlue

Highlights
----------
* **Hydra/OmegaConf** configuration (all parameters in *config.yaml*).
* **Multi‑GPU** via ``torch.nn.DataParallel`` (configurable).
* **Checkpoint‑resume** every epoch (model, optimiser, scheduler).
* **Qualitative epoch snapshot** – ground‑truth vs. predicted CadQuery code +
  point‑clouds as ASCII *.ply* for Windows viewers.
* **Validation & logging** – aggregated metrics written to *log.txt* (console
  + file).  Best + last checkpoints kept, others pruned.
* **Smoke vs. full mode** (`training.mode: smoke|full`) for rapid pipeline
  checks.

The script assumes that **dataset.py**, **model.py**, and **utils.py** implement
all helper classes exactly as defined elsewhere in the repo – especially that
``CadRecodeDataset`` returns ``points`` with shape *(N_pts,3)* (default 256)
and a *str* of CadQuery code, and that ``CADRecodeModel`` consumes these
without squeezing the point‑cloud: every input point becomes its **own token**.
"""
from __future__ import annotations

import os
import sys
import glob
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:  # Hydra not installed – fall back but warn
    raise RuntimeError("Hydra is required:  pip install hydra-core") from e

# local imports (package‑relative)
from cad_recode.dataset import CadRecodeDataset
from cad_recode.model   import CADRecodeModel
from cad_recode.utils   import (
    save_point_cloud,
    sample_points_on_shape,
    chamfer_distance,
)

# ---------------------------------------------------------------------------
#  Helper: logging setup (console + file)
# ---------------------------------------------------------------------------

def _init_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("cad_recode.train")
    logger.setLevel(logging.INFO)
    # purge default handlers (Jupyter duplicates otherwise)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                             "%Y-%m-%d %H:%M:%S")
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh  = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

# ---------------------------------------------------------------------------
#  Validation helper (fast quantitative pass)
# ---------------------------------------------------------------------------

def _validate(model: nn.Module,
              loader: DataLoader,
              device: torch.device) -> float:
    """Return mean loss over *loader* (no grad)."""
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for pts, code in loader:
            pts = pts.to(device, non_blocking=True)
            code_end = [s + "<|end|>" for s in code]
            out = model(pts, code=code_end, labels=code_end)
            loss_sum += out.loss.item() * pts.size(0)
            n += pts.size(0)
    model.train()
    return loss_sum / max(1, n)

# ---------------------------------------------------------------------------
#  Qualitative snapshot (GT vs. prediction) – returns dict with file paths
# ---------------------------------------------------------------------------

def _save_epoch_snapshot(model: nn.Module,
                         sample: tuple[torch.Tensor, str],
                         out_dir: Path,
                         device: torch.device,
                         max_tokens: int = 256) -> dict[str, str]:
    """Generate a single prediction → write code & PLYs → return paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pts, gt_code = sample
    pts = pts.unsqueeze(0).to(device)  # (1,N,3)

    # ------------------------------------------------------------------
    #  Run model.generate via helper prepare_prefix
    # ------------------------------------------------------------------
    with torch.no_grad():
        prefix_emb, attn = model.prepare_prefix(pts)
        gen_tok = model.decoder.generate(
            inputs_embeds        = prefix_emb,
            attention_mask       = attn,
            max_new_tokens       = max_tokens,
            num_beams            = 3,
            do_sample            = False,
            eos_token_id         = model.end_id,
        )[0]
        pred_code = model.tokenizer.decode(gen_tok, skip_special_tokens=False)

    # save codes --------------------------------------------------------
    gt_code_file   = out_dir / "ground_truth.py"
    pred_code_file = out_dir / "predicted.py"
    gt_code_file.write_text(gt_code + "\n")
    pred_code_file.write_text(pred_code + "\n")

    # save point‑clouds (PLY) ------------------------------------------
    # Generate dense PCs for nicer visual comparison (1024 pts)
    try:
        loc = {}
        exec(gt_code,  {"cq": __import__("cadquery").cq}, loc)
        solid_gt = loc.get("result") or loc.get("r") or loc.get("shape")
        if isinstance(solid_gt, __import__("cadquery").cq.Workplane):
            solid_gt = solid_gt.val()
        pc_gt = sample_points_on_shape(solid_gt, 1024)
        save_point_cloud(pc_gt, out_dir / "ground_truth.ply")
    except Exception as e:
        pc_gt = None

    try:
        loc = {}
        exec(pred_code, {"cq": __import__("cadquery").cq}, loc)
        solid_pr = loc.get("result") or loc.get("r") or loc.get("shape")
        if isinstance(solid_pr, __import__("cadquery").cq.Workplane):
            solid_pr = solid_pr.val()
        pc_pr = sample_points_on_shape(solid_pr, 1024)
        save_point_cloud(pc_pr, out_dir / "predicted.ply")
    except Exception as e:
        pc_pr = None

    # optional quick Chamfer for console
    if pc_gt is not None and pc_pr is not None:
        cd = chamfer_distance(pc_gt, pc_pr)
    else:
        cd = float("nan")

    logger = logging.getLogger("cad_recode.train")
    logger.info("=== Epoch Snapshot ===")
    logger.info("Ground Truth CadQuery Code:\n%s", gt_code)
    logger.info("Predicted CadQuery Code:\n%s", pred_code)

    return {
        "gt_code": str(gt_code_file),
        "pred_code": str(pred_code_file),
        "gt_ply": str(out_dir / "ground_truth.ply"),
        "pred_ply": str(out_dir / "predicted.ply"),
        "chamfer": cd,
    }

# ---------------------------------------------------------------------------
#  Main training entry point (Hydra)                                         
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):  # pylint: disable=too-many-locals,too-many-statements
        
    # ------------------------------------------------------------------
    #  0) Sanity Checks
    # ------------------------------------------------------------------
    # Dataset path check
    data_root = Path(cfg.data.path).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_root}")

    # Check subfolders exist
    for split in ["train", "val"]:
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

    # Seed
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)


    """Main training driver – invoked by CLI *or* imported by notebooks."""
    # ------------------------------------------------------------------
    #  1) Resolve paths/output folders
    # ------------------------------------------------------------------
    proj_root   = Path(__file__).resolve().parent.parent  # cad_recode_base/
    output_root = proj_root / "output"
    output_root.mkdir(exist_ok=True)

    # auto‑increment run_X
    existing = [int(p.name.split('_')[1]) for p in output_root.glob('run_*') if p.name.split('_')[1].isdigit()]
    run_idx  = max(existing, default=0) + 1
    run_dir  = output_root / f"run_{run_idx}"
    run_dir.mkdir()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()

    logger = _init_logger(run_dir / "log.txt")
    logger.info("🛠  New training run → %s", run_dir.name)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    #  2) Seed + device + (optional) multi‑GPU
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu > 1 and cfg.training.multi_gpu:
        logger.info("Using %d GPUs via DataParallel", n_gpu)
    else:
        logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    #  3) Dataset + DataLoaders
    # ------------------------------------------------------------------
    ds_train = CadRecodeDataset(cfg.data.path, split="train",
                                n_points   = cfg.data.n_points,
                                noise_std  = cfg.data.noise_std,
                                noise_prob = cfg.data.noise_prob)
    print(f"→ Resolved train path: {data_root / 'train'}\n")

    ds_val   = CadRecodeDataset(cfg.data.path, split="val",
                                n_points   = cfg.data.n_points,
                                noise_std  = 0.0, noise_prob = 0.0)
    print(f"→ Resolved val path: {data_root / 'val'}\n")

    # smoke‑mode trims dataset size for debugging
    if cfg.training.mode == "smoke":
        ds_train.files = ds_train.files[:8]
        ds_val.files   = ds_val.files[:4]
        logger.warning("Smoke mode: dataset truncated for quick test")

    dl_train = DataLoader(ds_train,
                          batch_size  = cfg.training.batch_size,
                          shuffle     = True,
                          num_workers = cfg.training.num_workers,
                          pin_memory  = True)
    dl_val   = DataLoader(ds_val,
                          batch_size  = cfg.training.batch_size,
                          shuffle     = False,
                          num_workers = max(cfg.training.num_workers // 2, 0),
                          pin_memory  = True)
    
    # Check for empty training dataset
    if len(dl_train.dataset) == 0:
        raise RuntimeError(
            f"Training dataset is empty.\n"
            f"→ Resolved path: {data_root / 'train'}\n"
            f"→ Check if the directory exists and contains .py CAD files."
        )

    # Check for empty validation dataset
    if len(dl_val.dataset) == 0:
        logger.warning(
            "⚠️ Validation dataset is empty.\n"
            f"→ Resolved path: {data_root / 'val'}\n"
            f"→ Check if the directory exists and contains .py CAD files."
        )


    # ------------------------------------------------------------------
    #  4) Model + optim + schedulers
    # ------------------------------------------------------------------
    model = CADRecodeModel(llm_name       = cfg.model.name,
                           freeze_decoder = cfg.model.freeze_decoder)
    if n_gpu > 1 and cfg.training.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr          = cfg.training.lr,
                      weight_decay= cfg.training.weight_decay)

    total_steps  = cfg.training.max_steps if cfg.training.max_steps > 0 else None
    warmup_steps = cfg.training.warmup_steps

    if total_steps is None:  # epoch‑based run → derive steps
        total_steps = cfg.training.max_epochs * len(dl_train)

    sch1 = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-6, total_iters=warmup_steps)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sch1, sch2], [warmup_steps])

    # resume? ----------------------------------------------------------------
    start_step = 0
    if cfg.training.resume_path:
        ckpt_path = Path(cfg.training.resume_path)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optim'])
            scheduler.load_state_dict(ckpt['sched'])
            start_step = ckpt['step'] + 1
            logger.info("Resumed from %s (step %d)", ckpt_path, start_step)
        else:
            logger.warning("Resume checkpoint %s not found – starting fresh", ckpt_path)

    # ------------------------------------------------------------------
    #  5) Training loop
    # ------------------------------------------------------------------
    model.train()
    step, best_val = start_step, float('inf')
    accum_steps = max(1, cfg.training.accumulation_steps)
    if accum_steps > 1 and isinstance(model, torch.nn.DataParallel):
        grad_ctx = model.no_sync
    else:
        grad_ctx = torch.enable_grad

    fixed_val_sample: Optional[tuple[torch.Tensor, str]] = None
    if len(ds_val) > 0:
        fixed_val_sample = ds_val[0]  # for snapshot after each epoch

    epoch = 0
    while step < total_steps:
        epoch += 1
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}", leave=False)
        for i, (pts, code_str) in enumerate(pbar):
            pts = pts.to(device, non_blocking=True)  # (B,256,3)
            code_end = [s + "<|end|>" for s in code_str]
            outputs  = model(pts, code=code_end, labels=code_end)
            loss     = outputs.loss / accum_steps

            with grad_ctx():
                loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=loss.item()*accum_steps, lr=f"{lr_now:.2e}")
                if step % cfg.logging.log_interval == 0:
                    logger.info("step %d/%d  loss=%.4f  lr=%e", step, total_steps,
                                loss.item()*accum_steps, lr_now)

                # validation & checkpoint every val_interval
                if step % cfg.training.val_interval == 0 or step >= total_steps:
                    val_loss = _validate(model, dl_val, device)
                    logger.info("→ val_loss=%.4f (step %d)", val_loss, step)

                    # snapshot qualitative output
                    snap_dir = ckpt_dir / f"snapshot_{step:07d}"
                    snap = _save_epoch_snapshot(model, fixed_val_sample, snap_dir,
                                                device) if fixed_val_sample else {}

                    # save checkpoint
                    ckpt_file = ckpt_dir / f"ckpt_{step:07d}.pt"
                    torch.save({
                        'model': (model.module.state_dict() if isinstance(model, nn.DataParallel)
                                  else model.state_dict()),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                        'step':  step,
                        'val_loss': val_loss,
                        'snapshot': snap,
                    }, ckpt_file)
                    logger.info("Checkpoint @ step %d saved → %s", step, ckpt_file.name)

                    # track best
                    if val_loss < best_val:
                        best_val, best_ckpt = val_loss, ckpt_file

                if step >= total_steps:
                    break
        if step >= total_steps or epoch >= cfg.training.max_epochs:
            break

    # Final test-time evaluation (optional)
    if best_ckpt and cfg.training.run_final_test:
        logger.info("Running final evaluation on test split using best checkpoint")

        # Import the CLI main from evaluate.py
        from cad_recode.evaluate import main as evaluate_main
        import argparse

        eval_args = argparse.Namespace(
            data_root=cfg.data.path,
            checkpoint=str(best_ckpt),
            split="test",
            llm=cfg.model.name,
            max_length=cfg.model.max_length if "max_length" in cfg.model else 256,
            num_candidates=cfg.model.num_candidates if "num_candidates" in cfg.model else 1,
            output_dir=str(run_dir),
            save_examples=5,
        )

        evaluate_main(eval_args)


    # ------------------------------------------------------------------
    #  6) Final housekeeping – keep only *best* + *last*
    # ------------------------------------------------------------------
    ckpts = sorted(ckpt_dir.glob('ckpt_*.pt'))
    last_ckpt = ckpts[-1] if ckpts else None
    for f in ckpts:
        if f not in {last_ckpt, best_ckpt}:
            try:
                f.unlink()
            except OSError:
                pass
    logger.info("Training done.  Best val=%.4f  total_steps=%d", best_val, step)


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        from omegaconf import OmegaConf
        this_file = Path(__file__).resolve()
        config_path = this_file.parent.parent / "config" / "config.yaml"
        cfg = OmegaConf.load(config_path)
        main(cfg)
    else:
        main()