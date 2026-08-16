cd /home/minhtri/Molecular_AD

export RUN_DIR="/home/minhtri/Molecular_AD/checkpoints/2026-08-14_15-45-21"

python post_analysis/extract_latents.py && \
python post_analysis/analyze_dci.py
python post_analysis/compute_index.py && \
python post_analysis/validate_index.py && \
python post_analysis/plot_latent_map.py