cd /home/minhtri/Molecular_AD

export RUN_DIR="/home/minhtri/Molecular_AD/checkpoints/2026-08-16_13-48-33"

# python post_analysis/extract_latents.py && \
# python post_analysis/analyze_dci.py && \
# python post_analysis/compute_index.py && \
# python post_analysis/validate_index.py && \
python post_analysis/plot_latent_map.py 
# python post_analysis/age_index.py