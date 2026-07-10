DATAPATH="/Users/wenming/green-bse-periodic/green-bse-periodic-bse-solver/example/BN_STO3G_hex"
BINPATH="/Users/wenming/green-mbpt/build"
BETA=1000

mpirun -n 4 $BINPATH/mbpt.exe \
    --mixing_type SIGMA_MIXING \
    --mixing_weight 0.60 \
    --input_file mean_field_input.h5 \
    --results_file scGW_sim_beta_$BETA.h5 \
    --dfintegral_hf_file $DATAPATH/df_hf_int \
    --dfintegral_file $DATAPATH/df_int \
    --itermax 250 \
    --threshold 1e-6 \
    --grid_file $DATAPATH/../H2_STO3G_grid/irgrid/1e5.h5 \
    --scf_type GW \
    --BETA $BETA
