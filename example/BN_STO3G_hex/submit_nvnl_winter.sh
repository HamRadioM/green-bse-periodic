DATAPATH="/Users/wenming/green-bse-periodic/green-bse-periodic-bse-solver/example/BN_STO3G_hex"
BETA=1000

# Wannier interpolation of F(k) + Sigma(tau,k) onto the G-M-K-G path, Dyson on
# the path, then Nevanlinna continuation of the SAO-orthogonalized G(iw) ->
# A(k, w) band structure.  Window +-1.5 a.u. (~ +-40 eV) covers the frontier
# valence/conduction bands; the B/N 1s cores (-200/-400 eV) are outside.
python $DATAPATH/nvnl_winter_analysis_2d.py \
    --beta $BETA \
    --dim 2 \
    --bandpath G M K G \
    --bandpts 121 \
    --input $DATAPATH/mean_field_input.h5 \
    --sim $DATAPATH/scGW_sim_beta_$BETA.h5 \
    --ir_file $DATAPATH/../H2_STO3G_grid/irgrid/1e5.h5 \
    --orth sao \
    --eta 0.005 \
    --n_real 1501 \
    --w_min -1.5 \
    --w_max 1.5 \
    --out nvnl_winter_out.h5

# traced A(k, w) heatmap along the path
python $DATAPATH/plot_nvnl_winter.py nvnl_winter_out.h5 --out nvnl_winter_bands
