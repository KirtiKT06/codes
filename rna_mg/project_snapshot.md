# Project snapshot

root: `/data/rna_electrostats`   code: `/home/feynman/projects/codes/rna_mg`


## 0a. Data tree (top level of --root)

```
bench_apbs_relaxed                     81 files      2MB   sub: chains, per_residue
bench_apbs_v2                         382 files     12MB   sub: chains, per_residue
bench_coulomb_v2                      382 files     12MB   sub: chains, per_residue
bench_out                             382 files      7MB   sub: chains, per_residue
centered_latent                        85 files    288MB   sub: cross_model, ernierna, rnabert, rnafm, splicebert
centered_latent_v2                    138 files    566MB   sub: cross_model, cross_model_gbm, ernierna, rnabert, rnafm, splicebert
centered_latent_v3                    275 files      3GB   sub: cross_model, ernierna, rnabert, rnafm, splicebert
characterization                       16 files     18KB   sub: ernierna, rnabert, rnafm, splicebert
clustering                              6 files      3MB   sub: mmseqs_work
controls                             1632 files     10GB   sub: family, random_init
controls_matched                      729 files      2GB   sub: analysis, family_t0.5_fold0, family_t0.5_fold1, family_t0.5_fold2, family_t0.5_fold3, family_t0.5_fold4, famrand_t0.5_fold0_s0, famrand_t0.5_fold1_s0, famrand_t0.5_fold2_s0, famrand_t0.5_fold3_s0, famrand_t0.5_fold4_s0, pretrained
coulomb_report                          1 files      2KB   sub: 
identity_ctl                           57 files      3MB   sub: A, B, C, out_A, out_B, out_C
latent_manifold                        20 files     22KB   sub: ernierna, rnabert, rnafm, splicebert
mg_full                               592 files      7MB   sub: labels_r0.79, per_residue
mg_pilot                               74 files      1MB   sub: labels_r0.79, per_residue
phi_res                             31604 files    175GB   sub: dssr_json, raw_pdb, work
preds                                 189 files     21MB   sub: 
preds_relaxed                          40 files      4MB   sub: 
priority_results                        2 files     57MB   sub: secondary_structure
priority_results_c1prime               56 files     27MB   sub: ernierna, rnabert, rnafm, splicebert
probe_all_models_v2                    41 files    166KB   sub: 
results_c1prime                         2 files    385KB   sub: composition
coulomb_baseline.csv.gz                57MB
coulomb_baseline_structure_summary.csv     39KB
dssr_controls.csv.gz                    3MB
dssr_controls_join_report.csv          31KB
structure_controls.csv.gz               7MB
```


## 0b. Code directory

```
analyze_failure.py                               153 lines
audit_extras.py                                  260 lines
build_dssr_controls.py                           416 lines
build_family_table.py                            182 lines
build_structure_controls.py                      136 lines
centered_latent_manifold.py                      785 lines
collect_results.py                               505 lines
combined_identity_structure_baseline.py          163 lines
compare_audit_runs.py                             65 lines
composition_position_baselines.py                219 lines
coulomb_residual_test.py                         407 lines
cross_model_latent_agreement.py                  262 lines
electrostatic_field_mapping.py                   360 lines
electrostatic_latent_manifold.py                 412 lines
filter_dataset.py                                 45 lines
latent_controls.py                               190 lines
list_long_chains.py                               12 lines
make_clean_subset.py                             199 lines
make_figures.py                                  508 lines
make_figures_extra.py                            260 lines
mg_sensitivity.py                                484 lines
mmseqs_cluster.py                                149 lines
patch_audit.py                                   148 lines
predicted_structure_benchmark.py                 633 lines
repair_pdb_ids.py                                 95 lines
representation_characterization.py               479 lines
rhofold_feature_hook.py                          146 lines
rna_electrostatics_pipeline.py                   843 lines
rna_electrostatics_pipeline_v2.py               1495 lines
rna_embedding_probe.py                           727 lines
rna_embedding_probe_v2.py                        734 lines
rna_probe_priority_experiments.py                632 lines
rna_probe_priority_experiments_extended.py      1302 lines
run_rhofold_on_fasta.py                           97 lines
secondary_structure_baseline.py                  211 lines
secondary_structure_baseline_modified.py         235 lines
structural_3d_baseline.py                        309 lines
README.md                                          19KB
REPORT.md                                          13KB
project_snapshot.md                               201KB
length_table.txt                                    8KB
results_bundle.txt                                 10KB
results_bundle2.txt                                 4KB
results_bundle3.txt                                 8KB
```


## 0c. Logs with 'Selected layer' (the layer is NOT stored in the audit CSVs)

```
none found -- layer per run unknown from files alone
```


## 1. Dataset / labels

```
labels file: /data/rna_electrostats/phi_res/all_residue_potentials.csv  columns: ['pdb_id', 'chain', 'resnum', 'resname', 'x', 'y', 'z', 'potential_mean_kT_e_4A', 'potential_std_kT_e_4A', 'n_grid_points_4A', 'potential_mean_kT_e_8A', 'potential_std_kT_e_8A', 'n_grid_points_8A', 'potential_mean_kT_e_12A', 'potential_std_kT_e_12A', 'n_grid_points_12A', 'potential_at_c1prime_kT_e', 'potential_at_phosphorus_kT_e', 'original_chain', 'original_resnum', 'insertion_code', 'charges_possibly_incomplete', 'had_modified_nucleotides', 'split', 'resolution_A', 'experimental_method', 'pb_solver']
residues: 696,204   structures: 1,873   chains: 4,342
solver per structure:
pb_solver
nonlinear              1812
linearized_fallback      61
residue types:
resname
RG     206886
RA     177674
RC     158643
RU     144318
RG5      1807
RC3      1355
RA3      1153
RU3      1010
structure_controls.csv.gz: 7MB
dssr_controls.csv.gz: 3MB
coulomb_baseline.csv.gz: 57MB
```


## 2. Raw probe, final held-out R^2 (best layer by val; rows probe == final_test)

```
                  label_column  ernierna  rnabert  rnafm  splicebert  identity_baseline
     potential_at_c1prime_kT_e     0.607    0.624  0.614       0.586              0.289
   potential_at_c1prime_kT_e_z     0.508    0.527  0.492       0.452              0.419
  potential_at_phosphorus_kT_e     0.675    0.580  0.684       0.663             -0.006
potential_at_phosphorus_kT_e_z     0.009    0.000  0.008       0.002              0.000
       potential_mean_kT_e_12A     0.405    0.266  0.343       0.286              0.005
     potential_mean_kT_e_12A_z     0.254    0.049  0.178       0.116              0.001
        potential_mean_kT_e_4A     0.330    0.178  0.282       0.224              0.041
      potential_mean_kT_e_4A_z     0.229    0.090  0.178       0.134              0.067
        potential_mean_kT_e_8A     0.381    0.225  0.310       0.266              0.011
      potential_mean_kT_e_8A_z     0.238    0.054  0.171       0.126              0.011
```


## 2b. Layer chosen per (target, model)

```
                  label_column  ernierna  rnabert  rnafm  splicebert
     potential_at_c1prime_kT_e         3        5      5           1
   potential_at_c1prime_kT_e_z        12        3     10           6
  potential_at_phosphorus_kT_e         3        5      2           1
potential_at_phosphorus_kT_e_z         9        6     12           4
       potential_mean_kT_e_12A        11        1      5           5
     potential_mean_kT_e_12A_z        10        0      8           2
        potential_mean_kT_e_4A        11        1     10           4
      potential_mean_kT_e_4A_z        11        3      8           1
        potential_mean_kT_e_8A        11        1      5           4
      potential_mean_kT_e_8A_z        10        0      8           2
```


(priority results /data/rna_electrostats/priority_results/secondary_structure: residue_level_structure_features.csv, secondary_structure_summary.csv)


(priority results /data/rna_electrostats/priority_results_c1prime/ernierna: p10_residual_after_21mer.csv, p11_electrostatic_twin_pairs.csv, p11_electrostatic_twin_summary.csv, p1_structure_scale_probe.csv, p1b_centered_rank_probe.csv, p1c_same_support_base_identity.csv, p2_kmer_baselines.csv, p3_cluster_split.csv, p4_mlp_on_best_layer.csv, p5_negative_controls.csv, p6_bootstrap_ci.csv, p7_composition_residualized.csv, p8_kmer_sweep_vs_centered.csv, p9_delta_r2_lm_plus_structure.csv)


(priority results /data/rna_electrostats/priority_results_c1prime/rnabert: p10_residual_after_21mer.csv, p11_electrostatic_twin_pairs.csv, p11_electrostatic_twin_summary.csv, p1_structure_scale_probe.csv, p1b_centered_rank_probe.csv, p1c_same_support_base_identity.csv, p2_kmer_baselines.csv, p3_cluster_split.csv, p4_mlp_on_best_layer.csv, p5_negative_controls.csv, p6_bootstrap_ci.csv, p7_composition_residualized.csv, p8_kmer_sweep_vs_centered.csv, p9_delta_r2_lm_plus_structure.csv)


(priority results /data/rna_electrostats/priority_results_c1prime/rnafm: p10_residual_after_21mer.csv, p11_electrostatic_twin_pairs.csv, p11_electrostatic_twin_summary.csv, p1_structure_scale_probe.csv, p1b_centered_rank_probe.csv, p1c_same_support_base_identity.csv, p2_kmer_baselines.csv, p3_cluster_split.csv, p4_mlp_on_best_layer.csv, p5_negative_controls.csv, p6_bootstrap_ci.csv, p7_composition_residualized.csv, p8_kmer_sweep_vs_centered.csv, p9_delta_r2_lm_plus_structure.csv)


(priority results /data/rna_electrostats/priority_results_c1prime/splicebert: p10_residual_after_21mer.csv, p11_electrostatic_twin_pairs.csv, p11_electrostatic_twin_summary.csv, p1_structure_scale_probe.csv, p1b_centered_rank_probe.csv, p1c_same_support_base_identity.csv, p2_kmer_baselines.csv, p3_cluster_split.csv, p4_mlp_on_best_layer.csv, p5_negative_controls.csv, p6_bootstrap_ci.csv, p7_composition_residualized.csv, p8_kmer_sweep_vs_centered.csv, p9_delta_r2_lm_plus_structure.csv)


## 3a. Single-target ridge ceiling per variant (pretrained, shared layer)

```
                      target      model  centered  centered_base_resid  centered_coulomb_resid  centered_ctx_gbm  centered_dssr_ctx_coulomb_gbm  centered_dssr_ctx_gbm  centered_dssr_gbm  centered_nuisance_resid  centered_ss_gbm  centered_struct_gbm
   local_minus_global_4A_12A   ernierna     0.166                0.096                   0.076             0.057                          0.025                  0.026              0.038                    0.096            0.064                0.049
   local_minus_global_4A_12A    rnabert     0.089                0.001                   0.005            -0.001                          0.002                  0.000             -0.000                    0.002           -0.000               -0.000
   local_minus_global_4A_12A      rnafm     0.135                0.065                   0.045             0.031                          0.017                  0.018              0.028                    0.065            0.044                0.034
   local_minus_global_4A_12A splicebert     0.114                0.039                   0.036             0.010                          0.006                  0.006              0.015                    0.039            0.028                0.021
   potential_at_c1prime_kT_e   ernierna     0.529                0.199                   0.139             0.090                          0.001                  0.015              0.020                    0.198            0.079                0.035
   potential_at_c1prime_kT_e    rnabert     0.567                0.050                   0.082            -0.008                         -0.002                 -0.007             -0.007                    0.050           -0.006               -0.007
   potential_at_c1prime_kT_e      rnafm     0.515                0.151                   0.109             0.052                          0.000                  0.013              0.016                    0.148            0.052                0.021
   potential_at_c1prime_kT_e splicebert     0.465                0.076                   0.077             0.013                         -0.002                 -0.002              0.001                    0.075            0.019                0.009
potential_at_phosphorus_kT_e   ernierna     0.004                0.004                  -0.000             0.001                         -0.001                 -0.000             -0.001                    0.003            0.001               -0.000
potential_at_phosphorus_kT_e    rnabert    -0.000                0.000                   0.000             0.000                         -0.000                 -0.000             -0.000                    0.000           -0.000               -0.000
potential_at_phosphorus_kT_e      rnafm     0.002                0.002                  -0.001             0.000                         -0.001                 -0.001             -0.001                    0.002            0.000               -0.001
potential_at_phosphorus_kT_e splicebert     0.001                0.001                   0.000            -0.000                         -0.001                 -0.001             -0.001                    0.001           -0.000               -0.001
     potential_mean_kT_e_12A   ernierna     0.312                0.310                   0.088             0.156                         -0.002                  0.045              0.061                    0.303            0.144                0.065
     potential_mean_kT_e_12A    rnabert     0.016                0.012                   0.005            -0.004                         -0.002                 -0.006             -0.006                    0.005           -0.003               -0.004
     potential_mean_kT_e_12A      rnafm     0.218                0.217                   0.055             0.095                          0.001                  0.023              0.036                    0.207            0.093                0.041
     potential_mean_kT_e_12A splicebert     0.106                0.104                   0.040             0.034                         -0.005                  0.014              0.021                    0.092            0.039                0.028
      potential_mean_kT_e_4A   ernierna     0.267                0.239                   0.091             0.125                          0.006                  0.023              0.034                    0.234            0.119                0.050
      potential_mean_kT_e_4A    rnabert     0.060                0.006                   0.005            -0.003                         -0.000                 -0.002             -0.002                    0.003           -0.002               -0.001
      potential_mean_kT_e_4A      rnafm     0.189                0.163                   0.052             0.071                          0.005                  0.017              0.025                    0.156            0.077                0.030
      potential_mean_kT_e_4A splicebert     0.112                0.078                   0.043             0.025                         -0.002                  0.007              0.015                    0.071            0.034                0.020
      potential_mean_kT_e_8A   ernierna     0.293                0.288                   0.096             0.149                         -0.000                  0.033              0.042                    0.281            0.141                0.056
      potential_mean_kT_e_8A    rnabert     0.019                0.009                   0.005            -0.004                         -0.002                 -0.004             -0.005                    0.003           -0.003               -0.003
      potential_mean_kT_e_8A      rnafm     0.202                0.198                   0.058             0.090                          0.004                  0.019              0.026                    0.190            0.092                0.035
      potential_mean_kT_e_8A splicebert     0.100                0.093                   0.046             0.029                         -0.007                  0.009              0.015                    0.083            0.037                0.021
```


## 3b. Variance of centered labels explained by controls alone (control_explained_r2.csv; pretrained runs)

```
                      target      model  centered_coulomb_resid  centered_ctx_gbm  centered_dssr_ctx_coulomb_gbm  centered_dssr_ctx_gbm  centered_dssr_gbm  centered_ss_gbm  centered_struct_gbm
   local_minus_global_4A_12A   ernierna                   0.575             0.186                          0.724                  0.453              0.431            0.150                0.354
   local_minus_global_4A_12A    rnabert                   0.597             0.195                          0.722                  0.447              0.429            0.173                0.353
   local_minus_global_4A_12A      rnafm                   0.575             0.188                          0.724                  0.452              0.433            0.149                0.355
   local_minus_global_4A_12A splicebert                   0.575             0.186                          0.724                  0.453              0.431            0.150                0.354
   potential_at_c1prime_kT_e   ernierna                   0.650             0.566                          0.796                  0.706              0.701            0.555                0.678
   potential_at_c1prime_kT_e    rnabert                   0.719             0.674                          0.838                  0.766              0.764            0.673                0.747
   potential_at_c1prime_kT_e      rnafm                   0.650             0.573                          0.792                  0.708              0.700            0.562                0.676
   potential_at_c1prime_kT_e splicebert                   0.650             0.566                          0.796                  0.706              0.701            0.555                0.678
potential_at_phosphorus_kT_e   ernierna                   0.013            -0.001                          0.014                  0.006              0.007            0.000                0.004
potential_at_phosphorus_kT_e    rnabert                   0.007            -0.002                          0.006                  0.000             -0.000           -0.002                0.001
potential_at_phosphorus_kT_e      rnafm                   0.013            -0.000                          0.015                  0.006              0.006            0.000                0.005
potential_at_phosphorus_kT_e splicebert                   0.013            -0.001                          0.014                  0.006              0.007            0.000                0.004
     potential_mean_kT_e_12A   ernierna                   0.607             0.244                          0.744                  0.580              0.579            0.227                0.517
     potential_mean_kT_e_12A    rnabert                   0.635             0.266                          0.783                  0.610              0.600            0.243                0.555
     potential_mean_kT_e_12A      rnafm                   0.606             0.263                          0.767                  0.579              0.570            0.219                0.507
     potential_mean_kT_e_12A splicebert                   0.607             0.244                          0.744                  0.580              0.579            0.227                0.517
      potential_mean_kT_e_4A   ernierna                   0.576             0.235                          0.747                  0.586              0.573            0.202                0.514
      potential_mean_kT_e_4A    rnabert                   0.593             0.246                          0.767                  0.581              0.571            0.220                0.518
      potential_mean_kT_e_4A      rnafm                   0.576             0.243                          0.755                  0.587              0.568            0.201                0.512
      potential_mean_kT_e_4A splicebert                   0.576             0.235                          0.747                  0.586              0.573            0.202                0.514
      potential_mean_kT_e_8A   ernierna                   0.576             0.251                          0.744                  0.615              0.603            0.214                0.548
      potential_mean_kT_e_8A    rnabert                   0.590             0.250                          0.767                  0.645              0.635            0.226                0.588
      potential_mean_kT_e_8A      rnafm                   0.576             0.254                          0.757                  0.619              0.599            0.212                0.541
      potential_mean_kT_e_8A splicebert                   0.576             0.251                          0.744                  0.615              0.603            0.214                0.548
```


## 3c. Nuisance-only baselines (linear variants; one model shown, they are identical across models)

```
                variant                       target  nuisance_only_test_r2
               centered    potential_at_c1prime_kT_e                  0.438
               centered potential_at_phosphorus_kT_e                  0.000
               centered       potential_mean_kT_e_4A                  0.046
               centered       potential_mean_kT_e_8A                  0.019
               centered      potential_mean_kT_e_12A                  0.015
               centered    local_minus_global_4A_12A                  0.082
centered_nuisance_resid    potential_at_c1prime_kT_e                  0.000
centered_nuisance_resid potential_at_phosphorus_kT_e                  0.000
centered_nuisance_resid       potential_mean_kT_e_4A                 -0.000
centered_nuisance_resid       potential_mean_kT_e_8A                 -0.000
centered_nuisance_resid      potential_mean_kT_e_12A                 -0.000
centered_nuisance_resid    local_minus_global_4A_12A                 -0.000
    centered_base_resid    potential_at_c1prime_kT_e                  0.003
    centered_base_resid potential_at_phosphorus_kT_e                  0.000
    centered_base_resid       potential_mean_kT_e_4A                  0.009
    centered_base_resid       potential_mean_kT_e_8A                  0.012
    centered_base_resid      potential_mean_kT_e_12A                  0.013
    centered_base_resid    local_minus_global_4A_12A                  0.001
```


## 4a. Which audit files were found

```
                   run      model  files
                family   ernierna      5
                family    rnabert      5
                family      rnafm      5
                family splicebert      5
   matched_family_t0.5   ernierna      5
   matched_family_t0.5      rnafm      5
matched_famrandom_t0.5   ernierna      5
matched_famrandom_t0.5      rnafm      5
    matched_pretrained   ernierna     17
    matched_pretrained    rnabert     17
    matched_pretrained      rnafm     17
    matched_pretrained splicebert     17
        matched_random   ernierna      3
        matched_random    rnabert      3
        matched_random      rnafm      3
        matched_random splicebert      3
            pretrained   ernierna      1
            pretrained    rnabert      1
            pretrained      rnafm      1
            pretrained splicebert      1
                random   ernierna      1
                random    rnabert      1
                random      rnafm      1
                random splicebert      1
```


### 4b. Variance partition: controls=controls_geom, micro  (single file: value [95% bootstrap CI]; family folds: mean [min..max])

```
                   target      model       component             pretrained                 random                      family         matched_family_t0.5      matched_famrandom_t0.5          matched_pretrained              matched_random
potential_at_c1prime_kT_e   ernierna       unique_lm +0.015 [+0.003,+0.022] -0.005 [-0.014,+0.003] -0.002 [-0.009..+0.003] n=5 -0.002 [-0.009..+0.005] n=5 -0.000 [-0.006..+0.007] n=5      +0.015 [+0.003,+0.022] -0.005 [-0.006..-0.004] n=3
potential_at_c1prime_kT_e   ernierna          shared +0.591 [+0.519,+0.701] +0.371 [+0.270,+0.520] +0.637 [+0.568..+0.743] n=5 +0.636 [+0.568..+0.743] n=5 +0.331 [+0.276..+0.410] n=5      +0.591 [+0.519,+0.701] +0.275 [+0.270..+0.278] n=3
potential_at_c1prime_kT_e   ernierna unique_controls +0.115 [+0.090,+0.135] +0.335 [+0.262,+0.382] +0.118 [+0.094..+0.132] n=5 +0.119 [+0.094..+0.132] n=5 +0.424 [+0.368..+0.470] n=5      +0.115 [+0.090,+0.135] +0.432 [+0.428..+0.436] n=3
potential_at_c1prime_kT_e   ernierna     unexplained +0.279 [+0.201,+0.330] +0.299 [+0.207,+0.361] +0.247 [+0.165..+0.302] n=5 +0.247 [+0.165..+0.302] n=5 +0.245 [+0.166..+0.294] n=5      +0.279 [+0.201,+0.330] +0.299 [+0.298..+0.299] n=3
potential_at_c1prime_kT_e    rnabert       unique_lm -0.002 [-0.004,+0.001] -0.002 [-0.004,+0.000] -0.002 [-0.006..+0.001] n=5                           -                           -      -0.002 [-0.004,+0.001] -0.002 [-0.003..-0.002] n=3
potential_at_c1prime_kT_e    rnabert          shared +0.628 [+0.542,+0.718] +0.632 [+0.539,+0.729] +0.661 [+0.613..+0.739] n=5                           -                           -      +0.628 [+0.542,+0.718] +0.634 [+0.633..+0.634] n=3
potential_at_c1prime_kT_e    rnabert unique_controls +0.138 [+0.098,+0.177] +0.134 [+0.087,+0.178] +0.116 [+0.090..+0.142] n=5                           -                           -      +0.138 [+0.098,+0.177] +0.132 [+0.132..+0.133] n=3
potential_at_c1prime_kT_e    rnabert     unexplained +0.236 [+0.180,+0.290] +0.236 [+0.181,+0.291] +0.226 [+0.174..+0.279] n=5                           -                           -      +0.236 [+0.180,+0.290] +0.237 [+0.236..+0.237] n=3
potential_at_c1prime_kT_e      rnafm       unique_lm +0.008 [-0.001,+0.014] -0.004 [-0.007,-0.000] -0.003 [-0.012..+0.001] n=5 -0.001 [-0.009..+0.006] n=5 -0.001 [-0.004..+0.001] n=5      +0.008 [-0.001,+0.014] -0.005 [-0.006..-0.004] n=3
potential_at_c1prime_kT_e      rnafm          shared +0.549 [+0.462,+0.674] +0.509 [+0.379,+0.691] +0.590 [+0.471..+0.738] n=5 +0.603 [+0.519..+0.701] n=5 +0.631 [+0.548..+0.730] n=5      +0.549 [+0.462,+0.674] +0.509 [+0.507..+0.510] n=3
potential_at_c1prime_kT_e      rnafm unique_controls +0.159 [+0.115,+0.189] +0.198 [+0.105,+0.275] +0.164 [+0.094..+0.247] n=5 +0.151 [+0.122..+0.200] n=5 +0.123 [+0.101..+0.171] n=5      +0.159 [+0.115,+0.189] +0.199 [+0.197..+0.201] n=3
potential_at_c1prime_kT_e      rnafm     unexplained +0.285 [+0.203,+0.339] +0.296 [+0.206,+0.356] +0.249 [+0.169..+0.301] n=5 +0.247 [+0.168..+0.301] n=5 +0.247 [+0.170..+0.295] n=5      +0.285 [+0.203,+0.339] +0.297 [+0.296..+0.298] n=3
potential_at_c1prime_kT_e splicebert       unique_lm +0.001 [-0.006,+0.007] -0.002 [-0.012,+0.006] -0.002 [-0.008..+0.003] n=5                           -                           -      +0.001 [-0.006,+0.007] -0.003 [-0.003..-0.003] n=3
potential_at_c1prime_kT_e splicebert          shared +0.413 [+0.336,+0.522] +0.508 [+0.383,+0.687] +0.494 [+0.436..+0.561] n=5                           -                           -      +0.413 [+0.336,+0.522] +0.509 [+0.508..+0.509] n=3
potential_at_c1prime_kT_e splicebert unique_controls +0.293 [+0.264,+0.319] +0.198 [+0.104,+0.273] +0.261 [+0.160..+0.304] n=5                           -                           -      +0.293 [+0.264,+0.319] +0.197 [+0.197..+0.198] n=3
potential_at_c1prime_kT_e splicebert     unexplained +0.293 [+0.205,+0.353] +0.296 [+0.205,+0.359] +0.247 [+0.166..+0.299] n=5                           -                           -      +0.293 [+0.205,+0.353] +0.297 [+0.296..+0.297] n=3
  potential_mean_kT_e_12A   ernierna       unique_lm +0.059 [+0.027,+0.085] -0.009 [-0.026,+0.015] +0.004 [-0.010..+0.037] n=5 +0.004 [-0.010..+0.039] n=5 -0.009 [-0.019..-0.000] n=5 +0.065 [+0.059..+0.085] n=9 -0.015 [-0.017..-0.014] n=3
  potential_mean_kT_e_12A   ernierna          shared +0.405 [+0.248,+0.511] +0.088 [-0.006,+0.166] +0.114 [+0.046..+0.174] n=5 +0.113 [+0.046..+0.168] n=5 -0.006 [-0.046..+0.021] n=5 +0.398 [+0.383..+0.407] n=9 +0.094 [+0.092..+0.096] n=3
  potential_mean_kT_e_12A   ernierna unique_controls +0.175 [+0.114,+0.291] +0.491 [+0.451,+0.561] +0.460 [+0.291..+0.527] n=5 +0.461 [+0.297..+0.527] n=5 +0.580 [+0.444..+0.655] n=5 +0.170 [+0.155..+0.178] n=9 +0.486 [+0.484..+0.488] n=3
  potential_mean_kT_e_12A   ernierna     unexplained +0.362 [+0.283,+0.435] +0.430 [+0.321,+0.502] +0.423 [+0.342..+0.498] n=5 +0.422 [+0.342..+0.496] n=5 +0.435 [+0.359..+0.546] n=5 +0.367 [+0.358..+0.392] n=9 +0.435 [+0.434..+0.437] n=3
  potential_mean_kT_e_12A    rnabert       unique_lm -0.002 [-0.013,+0.017] +0.008 [-0.004,+0.025] -0.007 [-0.012..+0.004] n=5                           -                           - +0.006 [-0.004..+0.045] n=9 -0.002 [-0.003..+0.001] n=3
  potential_mean_kT_e_12A    rnabert          shared +0.114 [+0.020,+0.190] +0.074 [+0.013,+0.135] +0.012 [-0.010..+0.038] n=5                           -                           - +0.120 [+0.114..+0.127] n=9 +0.085 [+0.084..+0.087] n=3
  potential_mean_kT_e_12A    rnabert unique_controls +0.496 [+0.414,+0.570] +0.535 [+0.459,+0.593] +0.572 [+0.472..+0.621] n=5                           -                           - +0.475 [+0.445..+0.496] n=9 +0.524 [+0.522..+0.525] n=3
  potential_mean_kT_e_12A    rnabert     unexplained +0.392 [+0.332,+0.510] +0.382 [+0.322,+0.502] +0.422 [+0.380..+0.495] n=5                           -                           - +0.398 [+0.386..+0.427] n=9 +0.392 [+0.390..+0.393] n=3
  potential_mean_kT_e_12A      rnafm       unique_lm +0.047 [+0.006,+0.062] -0.009 [-0.014,+0.004] -0.001 [-0.012..+0.022] n=5 +0.006 [-0.012..+0.042] n=5 -0.003 [-0.010..+0.003] n=5 +0.054 [+0.045..+0.068] n=9 -0.008 [-0.016..-0.002] n=3
  potential_mean_kT_e_12A      rnafm          shared +0.360 [+0.166,+0.467] +0.088 [+0.035,+0.180] +0.082 [+0.035..+0.145] n=5 +0.096 [+0.029..+0.182] n=5 -0.007 [-0.075..+0.030] n=5 +0.355 [+0.346..+0.366] n=9 +0.085 [+0.078..+0.094] n=3
  potential_mean_kT_e_12A      rnafm unique_controls +0.219 [+0.168,+0.349] +0.491 [+0.424,+0.552] +0.487 [+0.343..+0.551] n=5 +0.473 [+0.334..+0.557] n=5 +0.577 [+0.437..+0.661] n=5 +0.209 [+0.196..+0.219] n=9 +0.494 [+0.485..+0.501] n=3
  potential_mean_kT_e_12A      rnafm     unexplained +0.374 [+0.295,+0.456] +0.430 [+0.323,+0.514] +0.431 [+0.342..+0.519] n=5 +0.425 [+0.336..+0.498] n=5 +0.434 [+0.341..+0.542] n=5 +0.382 [+0.373..+0.402] n=9 +0.429 [+0.423..+0.437] n=3
  potential_mean_kT_e_12A splicebert       unique_lm +0.014 [-0.001,+0.030] -0.000 [-0.006,+0.013] -0.003 [-0.010..+0.011] n=5                           -                           - +0.023 [+0.009..+0.050] n=9 +0.000 [-0.001..+0.002] n=3
  potential_mean_kT_e_12A splicebert          shared +0.206 [+0.076,+0.351] +0.079 [+0.020,+0.167] +0.040 [-0.005..+0.102] n=5                           -                           - +0.206 [+0.188..+0.217] n=9 +0.077 [+0.075..+0.080] n=3
  potential_mean_kT_e_12A splicebert unique_controls +0.374 [+0.276,+0.460] +0.501 [+0.442,+0.555] +0.534 [+0.363..+0.614] n=5                           -                           - +0.362 [+0.339..+0.374] n=9 +0.503 [+0.500..+0.505] n=3
  potential_mean_kT_e_12A splicebert     unexplained +0.406 [+0.312,+0.480] +0.420 [+0.322,+0.497] +0.430 [+0.345..+0.524] n=5                           -                           - +0.409 [+0.398..+0.432] n=9 +0.420 [+0.418..+0.421] n=3
   potential_mean_kT_e_4A   ernierna       unique_lm +0.037 [+0.017,+0.045] -0.010 [-0.016,+0.002] -0.005 [-0.010..+0.015] n=5 -0.004 [-0.010..+0.021] n=5 -0.009 [-0.016..-0.001] n=5 +0.040 [+0.031..+0.068] n=9 -0.011 [-0.014..-0.008] n=3
   potential_mean_kT_e_4A   ernierna          shared +0.366 [+0.253,+0.427] +0.080 [+0.029,+0.131] +0.159 [+0.105..+0.189] n=5 +0.157 [+0.105..+0.184] n=5 +0.007 [-0.024..+0.041] n=5 +0.361 [+0.341..+0.369] n=9 +0.075 [+0.073..+0.076] n=3
   potential_mean_kT_e_4A   ernierna unique_controls +0.219 [+0.181,+0.282] +0.506 [+0.469,+0.524] +0.380 [+0.324..+0.402] n=5 +0.382 [+0.335..+0.402] n=5 +0.531 [+0.490..+0.576] n=5 +0.216 [+0.204..+0.220] n=9 +0.511 [+0.509..+0.513] n=3
   potential_mean_kT_e_4A   ernierna     unexplained +0.377 [+0.339,+0.451] +0.424 [+0.382,+0.485] +0.467 [+0.442..+0.503] n=5 +0.465 [+0.442..+0.503] n=5 +0.471 [+0.449..+0.504] n=5 +0.383 [+0.375..+0.403] n=9 +0.426 [+0.423..+0.428] n=3
   potential_mean_kT_e_4A    rnabert       unique_lm +0.003 [-0.007,+0.016] -0.001 [-0.006,+0.008] -0.006 [-0.015..+0.002] n=5                           -                           - +0.003 [-0.009..+0.035] n=9 -0.002 [-0.003..-0.002] n=3
   potential_mean_kT_e_4A    rnabert          shared +0.135 [+0.068,+0.192] +0.116 [+0.068,+0.158] +0.072 [+0.047..+0.094] n=5                           -                           - +0.145 [+0.135..+0.151] n=9 +0.118 [+0.117..+0.119] n=3
   potential_mean_kT_e_4A    rnabert unique_controls +0.446 [+0.395,+0.497] +0.465 [+0.418,+0.516] +0.469 [+0.450..+0.514] n=5                           -                           - +0.426 [+0.393..+0.446] n=9 +0.463 [+0.462..+0.464] n=3
   potential_mean_kT_e_4A    rnabert     unexplained +0.417 [+0.370,+0.483] +0.420 [+0.367,+0.493] +0.465 [+0.445..+0.496] n=5                           -                           - +0.426 [+0.413..+0.456] n=9 +0.421 [+0.421..+0.422] n=3
   potential_mean_kT_e_4A      rnafm       unique_lm +0.020 [+0.004,+0.027] -0.009 [-0.019,+0.006] -0.006 [-0.013..+0.006] n=5 -0.004 [-0.012..+0.013] n=5 -0.008 [-0.011..-0.004] n=5 +0.031 [+0.020..+0.051] n=9 -0.005 [-0.008..-0.002] n=3
   potential_mean_kT_e_4A      rnafm          shared +0.328 [+0.187,+0.405] +0.103 [+0.060,+0.162] +0.128 [+0.092..+0.188] n=5 +0.138 [+0.092..+0.199] n=5 +0.062 [+0.046..+0.086] n=5 +0.314 [+0.299..+0.328] n=9 +0.098 [+0.095..+0.102] n=3
   potential_mean_kT_e_4A      rnafm unique_controls +0.259 [+0.209,+0.343] +0.484 [+0.436,+0.514] +0.407 [+0.378..+0.443] n=5 +0.397 [+0.366..+0.432] n=5 +0.473 [+0.443..+0.520] n=5 +0.257 [+0.248..+0.265] n=9 +0.488 [+0.484..+0.492] n=3
   potential_mean_kT_e_4A      rnafm     unexplained +0.394 [+0.357,+0.472] +0.422 [+0.382,+0.479] +0.471 [+0.441..+0.505] n=5 +0.469 [+0.437..+0.505] n=5 +0.472 [+0.439..+0.506] n=5 +0.397 [+0.391..+0.416] n=9 +0.419 [+0.415..+0.422] n=3
   potential_mean_kT_e_4A splicebert       unique_lm +0.008 [+0.001,+0.018] -0.008 [-0.012,-0.004] -0.010 [-0.016..-0.000] n=5                           -                           - +0.009 [+0.002..+0.036] n=9 -0.004 [-0.004..-0.003] n=3
   potential_mean_kT_e_4A splicebert          shared +0.195 [+0.105,+0.279] +0.101 [+0.057,+0.172] +0.087 [+0.056..+0.133] n=5                           -                           - +0.204 [+0.192..+0.213] n=9 +0.095 [+0.094..+0.097] n=3
   potential_mean_kT_e_4A splicebert unique_controls +0.390 [+0.323,+0.430] +0.485 [+0.431,+0.517] +0.451 [+0.381..+0.488] n=5                           -                           - +0.375 [+0.355..+0.390] n=9 +0.490 [+0.489..+0.491] n=3
   potential_mean_kT_e_4A splicebert     unexplained +0.407 [+0.368,+0.467] +0.423 [+0.391,+0.481] +0.472 [+0.443..+0.510] n=5                           -                           - +0.412 [+0.400..+0.430] n=9 +0.418 [+0.418..+0.418] n=3
```


### 4b. Variance partition: controls=controls_geom, macro  (single file: value [95% bootstrap CI]; family folds: mean [min..max])

```
                   target      model       component             pretrained                 random                      family         matched_family_t0.5      matched_famrandom_t0.5          matched_pretrained              matched_random
potential_at_c1prime_kT_e   ernierna       unique_lm -0.001 [-0.004,+0.002] +0.000 [-0.001,+0.002] -0.003 [-0.004..-0.002] n=5 -0.003 [-0.004..-0.001] n=5 -0.001 [-0.002..+0.001] n=5      -0.001 [-0.004,+0.002] +0.001 [+0.000..+0.002] n=3
potential_at_c1prime_kT_e   ernierna          shared +0.668 [+0.613,+0.719] +0.694 [+0.649,+0.731] +0.699 [+0.669..+0.728] n=5 +0.698 [+0.669..+0.728] n=5 +0.520 [+0.487..+0.564] n=5      +0.668 [+0.613,+0.719] +0.578 [+0.564..+0.592] n=3
potential_at_c1prime_kT_e   ernierna unique_controls +0.171 [+0.126,+0.222] +0.145 [+0.124,+0.168] +0.153 [+0.101..+0.174] n=5 +0.153 [+0.105..+0.174] n=5 +0.332 [+0.293..+0.378] n=5      +0.171 [+0.126,+0.222] +0.260 [+0.247..+0.274] n=3
potential_at_c1prime_kT_e   ernierna     unexplained +0.163 [+0.138,+0.192] +0.161 [+0.136,+0.191] +0.152 [+0.127..+0.176] n=5 +0.151 [+0.127..+0.174] n=5 +0.149 [+0.124..+0.176] n=5      +0.163 [+0.138,+0.192] +0.160 [+0.160..+0.161] n=3
potential_at_c1prime_kT_e    rnabert       unique_lm -0.001 [-0.002,+0.001] -0.001 [-0.003,+0.001] -0.002 [-0.005..-0.001] n=5                           -                           -      -0.001 [-0.002,+0.001] -0.001 [-0.002..-0.001] n=3
potential_at_c1prime_kT_e    rnabert          shared +0.762 [+0.729,+0.791] +0.794 [+0.759,+0.822] +0.758 [+0.714..+0.784] n=5                           -                           -      +0.762 [+0.729,+0.791] +0.797 [+0.796..+0.798] n=3
potential_at_c1prime_kT_e    rnabert unique_controls +0.085 [+0.072,+0.098] +0.053 [+0.041,+0.067] +0.098 [+0.082..+0.116] n=5                           -                           -      +0.085 [+0.072,+0.098] +0.050 [+0.049..+0.051] n=3
potential_at_c1prime_kT_e    rnabert     unexplained +0.154 [+0.131,+0.182] +0.154 [+0.131,+0.184] +0.146 [+0.136..+0.174] n=5                           -                           -      +0.154 [+0.131,+0.182] +0.154 [+0.154..+0.155] n=3
potential_at_c1prime_kT_e      rnafm       unique_lm -0.002 [-0.005,+0.001] -0.003 [-0.006,-0.000] -0.003 [-0.004..-0.001] n=5 -0.002 [-0.004..-0.000] n=5 -0.002 [-0.003..-0.000] n=5      -0.002 [-0.005,+0.001] -0.001 [-0.003..+0.000] n=3
potential_at_c1prime_kT_e      rnafm          shared +0.747 [+0.709,+0.779] +0.786 [+0.749,+0.818] +0.723 [+0.603..+0.802] n=5 +0.738 [+0.666..+0.774] n=5 +0.781 [+0.714..+0.815] n=5      +0.747 [+0.709,+0.779] +0.788 [+0.783..+0.795] n=3
potential_at_c1prime_kT_e      rnafm unique_controls +0.094 [+0.077,+0.111] +0.054 [+0.039,+0.070] +0.131 [+0.071..+0.223] n=5 +0.116 [+0.097..+0.160] n=5 +0.073 [+0.056..+0.113] n=5      +0.094 [+0.077,+0.111] +0.053 [+0.046..+0.058] n=3
potential_at_c1prime_kT_e      rnafm     unexplained +0.161 [+0.136,+0.191] +0.162 [+0.137,+0.193] +0.148 [+0.129..+0.174] n=5 +0.148 [+0.129..+0.173] n=5 +0.148 [+0.128..+0.176] n=5      +0.161 [+0.136,+0.191] +0.161 [+0.159..+0.163] n=3
potential_at_c1prime_kT_e splicebert       unique_lm -0.000 [-0.002,+0.001] +0.001 [-0.002,+0.004] -0.001 [-0.002..+0.000] n=5                           -                           -      -0.000 [-0.002,+0.001] -0.000 [-0.002..+0.002] n=3
potential_at_c1prime_kT_e splicebert          shared +0.605 [+0.572,+0.635] +0.795 [+0.756,+0.826] +0.617 [+0.580..+0.712] n=5                           -                           -      +0.605 [+0.572,+0.635] +0.794 [+0.791..+0.795] n=3
potential_at_c1prime_kT_e splicebert unique_controls +0.234 [+0.215,+0.255] +0.043 [+0.027,+0.062] +0.235 [+0.115..+0.284] n=5                           -                           -      +0.234 [+0.215,+0.255] +0.045 [+0.044..+0.047] n=3
potential_at_c1prime_kT_e splicebert     unexplained +0.162 [+0.136,+0.192] +0.161 [+0.137,+0.191] +0.150 [+0.125..+0.174] n=5                           -                           -      +0.162 [+0.136,+0.192] +0.162 [+0.160..+0.163] n=3
  potential_mean_kT_e_12A   ernierna       unique_lm +0.009 [-0.013,+0.027] -0.010 [-0.017,-0.002] -0.002 [-0.014..+0.010] n=5 -0.002 [-0.014..+0.011] n=5 -0.005 [-0.012..+0.001] n=5 +0.028 [+0.009..+0.107] n=9 -0.009 [-0.010..-0.008] n=3
  potential_mean_kT_e_12A   ernierna          shared +0.159 [+0.042,+0.244] +0.091 [+0.053,+0.123] +0.068 [+0.035..+0.137] n=5 +0.071 [+0.035..+0.151] n=5 +0.063 [+0.014..+0.113] n=5 +0.146 [+0.095..+0.160] n=9 +0.095 [+0.091..+0.097] n=3
  potential_mean_kT_e_12A   ernierna unique_controls +0.426 [+0.339,+0.539] +0.494 [+0.462,+0.533] +0.538 [+0.438..+0.578] n=5 +0.535 [+0.423..+0.578] n=5 +0.543 [+0.504..+0.593] n=5 +0.389 [+0.311..+0.426] n=9 +0.491 [+0.488..+0.494] n=3
  potential_mean_kT_e_12A   ernierna     unexplained +0.405 [+0.376,+0.445] +0.425 [+0.386,+0.467] +0.396 [+0.346..+0.415] n=5 +0.396 [+0.346..+0.414] n=5 +0.399 [+0.355..+0.438] n=5 +0.437 [+0.405..+0.520] n=9 +0.424 [+0.422..+0.425] n=3
  potential_mean_kT_e_12A    rnabert       unique_lm -0.001 [-0.008,+0.008] +0.004 [-0.002,+0.012] -0.005 [-0.009..-0.003] n=5                           -                           - +0.027 [-0.004..+0.136] n=9 -0.000 [-0.001..+0.001] n=3
  potential_mean_kT_e_12A    rnabert          shared +0.060 [+0.004,+0.102] +0.070 [+0.031,+0.100] +0.028 [-0.006..+0.054] n=5                           -                           - +0.068 [+0.060..+0.081] n=9 +0.073 [+0.071..+0.077] n=3
  potential_mean_kT_e_12A    rnabert unique_controls +0.544 [+0.506,+0.588] +0.534 [+0.497,+0.567] +0.593 [+0.530..+0.649] n=5                           -                           - +0.486 [+0.385..+0.544] n=9 +0.531 [+0.527..+0.533] n=3
  potential_mean_kT_e_12A    rnabert     unexplained +0.397 [+0.359,+0.447] +0.392 [+0.353,+0.443] +0.384 [+0.359..+0.419] n=5                           -                           - +0.419 [+0.381..+0.506] n=9 +0.396 [+0.395..+0.397] n=3
  potential_mean_kT_e_12A      rnafm       unique_lm +0.006 [-0.016,+0.023] -0.007 [-0.011,-0.002] +0.000 [-0.005..+0.012] n=5 +0.003 [-0.005..+0.013] n=5 -0.001 [-0.007..+0.007] n=5 +0.031 [+0.006..+0.110] n=9 -0.009 [-0.013..-0.006] n=3
  potential_mean_kT_e_12A      rnafm          shared +0.173 [+0.070,+0.248] +0.057 [+0.019,+0.090] +0.083 [+0.034..+0.148] n=5 +0.091 [+0.041..+0.147] n=5 +0.009 [-0.066..+0.054] n=5 +0.160 [+0.128..+0.174] n=9 +0.055 [+0.051..+0.058] n=3
  potential_mean_kT_e_12A      rnafm unique_controls +0.414 [+0.330,+0.512] +0.530 [+0.496,+0.568] +0.520 [+0.427..+0.561] n=5 +0.512 [+0.428..+0.554] n=5 +0.593 [+0.536..+0.661] n=5 +0.372 [+0.311..+0.414] n=9 +0.532 [+0.529..+0.535] n=3
  potential_mean_kT_e_12A      rnafm     unexplained +0.408 [+0.377,+0.453] +0.421 [+0.379,+0.466] +0.397 [+0.349..+0.413] n=5 +0.395 [+0.345..+0.412] n=5 +0.398 [+0.347..+0.429] n=5 +0.437 [+0.408..+0.509] n=9 +0.422 [+0.419..+0.427] n=3
  potential_mean_kT_e_12A splicebert       unique_lm -0.003 [-0.013,+0.004] -0.007 [-0.014,-0.002] -0.004 [-0.015..+0.008] n=5                           -                           - +0.021 [-0.007..+0.106] n=9 -0.000 [-0.002..+0.001] n=3
  potential_mean_kT_e_12A splicebert          shared +0.096 [+0.028,+0.146] +0.064 [+0.031,+0.096] +0.061 [+0.018..+0.137] n=5                           -                           - +0.104 [+0.093..+0.115] n=9 +0.057 [+0.055..+0.059] n=3
  potential_mean_kT_e_12A splicebert unique_controls +0.489 [+0.440,+0.551] +0.521 [+0.491,+0.556] +0.546 [+0.438..+0.589] n=5                           -                           - +0.436 [+0.355..+0.489] n=9 +0.528 [+0.527..+0.530] n=3
  potential_mean_kT_e_12A splicebert     unexplained +0.418 [+0.382,+0.458] +0.422 [+0.384,+0.463] +0.398 [+0.358..+0.417] n=5                           -                           - +0.439 [+0.409..+0.517] n=9 +0.415 [+0.414..+0.417] n=3
   potential_mean_kT_e_4A   ernierna       unique_lm -0.004 [-0.015,+0.007] -0.002 [-0.007,+0.004] -0.009 [-0.015..-0.005] n=5 -0.008 [-0.015..-0.005] n=5 -0.007 [-0.015..+0.001] n=5 +0.009 [-0.010..+0.073] n=9 -0.005 [-0.007..-0.004] n=3
   potential_mean_kT_e_4A   ernierna          shared +0.197 [+0.155,+0.240] +0.108 [+0.085,+0.130] +0.149 [+0.126..+0.202] n=5 +0.148 [+0.126..+0.196] n=5 +0.054 [+0.022..+0.085] n=5 +0.182 [+0.127..+0.197] n=9 +0.082 [+0.077..+0.085] n=3
   potential_mean_kT_e_4A   ernierna unique_controls +0.332 [+0.270,+0.389] +0.421 [+0.364,+0.465] +0.386 [+0.364..+0.405] n=5 +0.387 [+0.364..+0.405] n=5 +0.481 [+0.439..+0.513] n=5 +0.318 [+0.270..+0.332] n=9 +0.447 [+0.444..+0.452] n=3
   potential_mean_kT_e_4A   ernierna     unexplained +0.475 [+0.426,+0.541] +0.473 [+0.429,+0.533] +0.474 [+0.436..+0.493] n=5 +0.473 [+0.431..+0.493] n=5 +0.472 [+0.440..+0.483] n=5 +0.491 [+0.474..+0.533] n=9 +0.476 [+0.475..+0.478] n=3
   potential_mean_kT_e_4A    rnabert       unique_lm +0.004 [-0.007,+0.017] +0.005 [-0.004,+0.015] -0.006 [-0.015..+0.003] n=5                           -                           - +0.015 [-0.006..+0.085] n=9 -0.004 [-0.009..+0.001] n=3
   potential_mean_kT_e_4A    rnabert          shared +0.106 [+0.077,+0.132] +0.122 [+0.097,+0.145] +0.085 [+0.060..+0.106] n=5                           -                           - +0.118 [+0.106..+0.134] n=9 +0.130 [+0.126..+0.136] n=3
   potential_mean_kT_e_4A    rnabert unique_controls +0.425 [+0.380,+0.464] +0.409 [+0.361,+0.445] +0.452 [+0.414..+0.490] n=5                           -                           - +0.386 [+0.311..+0.425] n=9 +0.401 [+0.395..+0.405] n=3
   potential_mean_kT_e_4A    rnabert     unexplained +0.465 [+0.423,+0.514] +0.464 [+0.422,+0.513] +0.469 [+0.427..+0.494] n=5                           -                           - +0.481 [+0.448..+0.545] n=9 +0.473 [+0.468..+0.478] n=3
   potential_mean_kT_e_4A      rnafm       unique_lm -0.002 [-0.012,+0.006] -0.009 [-0.021,+0.001] -0.006 [-0.011..+0.001] n=5 -0.006 [-0.010..-0.001] n=5 -0.007 [-0.009..-0.004] n=5 +0.014 [-0.006..+0.076] n=9 -0.004 [-0.006..-0.003] n=3
   potential_mean_kT_e_4A      rnafm          shared +0.180 [+0.134,+0.227] +0.117 [+0.094,+0.142] +0.143 [+0.104..+0.179] n=5 +0.148 [+0.123..+0.183] n=5 +0.089 [+0.072..+0.125] n=5 +0.175 [+0.153..+0.192] n=9 +0.106 [+0.102..+0.111] n=3
   potential_mean_kT_e_4A      rnafm unique_controls +0.351 [+0.297,+0.403] +0.414 [+0.348,+0.462] +0.389 [+0.352..+0.411] n=5 +0.383 [+0.347..+0.418] n=5 +0.442 [+0.404..+0.489] n=5 +0.321 [+0.288..+0.351] n=9 +0.425 [+0.420..+0.429] n=3
   potential_mean_kT_e_4A      rnafm     unexplained +0.471 [+0.427,+0.530] +0.478 [+0.431,+0.545] +0.474 [+0.428..+0.502] n=5 +0.475 [+0.431..+0.502] n=5 +0.475 [+0.436..+0.502] n=5 +0.490 [+0.471..+0.535] n=9 +0.473 [+0.472..+0.475] n=3
   potential_mean_kT_e_4A splicebert       unique_lm -0.000 [-0.006,+0.006] -0.005 [-0.010,-0.000] -0.009 [-0.018..-0.005] n=5                           -                           - +0.013 [-0.010..+0.073] n=9 -0.006 [-0.013..+0.001] n=3
   potential_mean_kT_e_4A splicebert          shared +0.128 [+0.095,+0.156] +0.120 [+0.093,+0.144] +0.110 [+0.067..+0.193] n=5                           -                           - +0.132 [+0.110..+0.147] n=9 +0.121 [+0.112..+0.132] n=3
   potential_mean_kT_e_4A splicebert unique_controls +0.401 [+0.341,+0.450] +0.409 [+0.350,+0.455] +0.425 [+0.381..+0.467] n=5                           -                           - +0.370 [+0.323..+0.401] n=9 +0.408 [+0.397..+0.417] n=3
   potential_mean_kT_e_4A splicebert     unexplained +0.471 [+0.427,+0.534] +0.476 [+0.432,+0.538] +0.474 [+0.433..+0.496] n=5                           -                           - +0.486 [+0.462..+0.526] n=9 +0.477 [+0.470..+0.484] n=3
```


### 4b. Variance partition: controls=controls_geom_coulomb, micro  (single file: value [95% bootstrap CI]; family folds: mean [min..max])

```
                   target      model       component             pretrained                 random                      family         matched_family_t0.5      matched_famrandom_t0.5          matched_pretrained              matched_random
potential_at_c1prime_kT_e   ernierna       unique_lm +0.003 [-0.003,+0.008] -0.002 [-0.007,+0.002] -0.003 [-0.006..+0.000] n=5 -0.003 [-0.006..-0.000] n=5 -0.002 [-0.004..+0.000] n=5      +0.003 [-0.003,+0.008] -0.004 [-0.007..+0.001] n=3
potential_at_c1prime_kT_e   ernierna          shared +0.603 [+0.535,+0.705] +0.368 [+0.265,+0.518] +0.637 [+0.574..+0.742] n=5 +0.637 [+0.574..+0.742] n=5 +0.332 [+0.282..+0.408] n=5      +0.603 [+0.535,+0.705] +0.273 [+0.270..+0.280] n=3
potential_at_c1prime_kT_e   ernierna unique_controls +0.193 [+0.135,+0.267] +0.428 [+0.310,+0.534] +0.201 [+0.138..+0.243] n=5 +0.201 [+0.138..+0.243] n=5 +0.506 [+0.471..+0.545] n=5      +0.193 [+0.135,+0.267] +0.522 [+0.515..+0.526] n=3
potential_at_c1prime_kT_e   ernierna     unexplained +0.202 [+0.143,+0.265] +0.206 [+0.145,+0.272] +0.164 [+0.121..+0.189] n=5 +0.164 [+0.121..+0.189] n=5 +0.163 [+0.120..+0.188] n=5      +0.202 [+0.143,+0.265] +0.208 [+0.204..+0.211] n=3
potential_at_c1prime_kT_e    rnabert       unique_lm -0.002 [-0.004,-0.001] -0.001 [-0.003,-0.000] -0.002 [-0.005..+0.001] n=5                           -                           -      -0.002 [-0.004,-0.001] -0.003 [-0.004..-0.003] n=3
potential_at_c1prime_kT_e    rnabert          shared +0.629 [+0.542,+0.720] +0.631 [+0.538,+0.729] +0.660 [+0.611..+0.739] n=5                           -                           -      +0.629 [+0.542,+0.720] +0.634 [+0.634..+0.635] n=3
potential_at_c1prime_kT_e    rnabert unique_controls +0.209 [+0.148,+0.273] +0.206 [+0.139,+0.278] +0.191 [+0.139..+0.222] n=5                           -                           -      +0.209 [+0.148,+0.273] +0.203 [+0.203..+0.204] n=3
potential_at_c1prime_kT_e    rnabert     unexplained +0.164 [+0.129,+0.203] +0.163 [+0.129,+0.202] +0.151 [+0.125..+0.169] n=5                           -                           -      +0.164 [+0.129,+0.203] +0.165 [+0.165..+0.166] n=3
potential_at_c1prime_kT_e      rnafm       unique_lm -0.003 [-0.009,+0.002] -0.003 [-0.007,+0.001] -0.003 [-0.006..+0.000] n=5 -0.003 [-0.004..-0.002] n=5 -0.002 [-0.003..-0.001] n=5      -0.003 [-0.009,+0.002] -0.001 [-0.003..-0.001] n=3
potential_at_c1prime_kT_e      rnafm          shared +0.560 [+0.474,+0.683] +0.508 [+0.377,+0.690] +0.590 [+0.473..+0.739] n=5 +0.604 [+0.527..+0.704] n=5 +0.632 [+0.549..+0.731] n=5      +0.560 [+0.474,+0.683] +0.505 [+0.503..+0.507] n=3
potential_at_c1prime_kT_e      rnafm unique_controls +0.232 [+0.150,+0.314] +0.283 [+0.158,+0.417] +0.250 [+0.141..+0.358] n=5 +0.236 [+0.177..+0.303] n=5 +0.208 [+0.150..+0.281] n=5      +0.232 [+0.150,+0.314] +0.286 [+0.285..+0.288] n=3
potential_at_c1prime_kT_e      rnafm     unexplained +0.211 [+0.145,+0.281] +0.211 [+0.147,+0.279] +0.163 [+0.121..+0.192] n=5 +0.163 [+0.121..+0.192] n=5 +0.162 [+0.120..+0.189] n=5      +0.211 [+0.145,+0.281] +0.210 [+0.209..+0.211] n=3
potential_at_c1prime_kT_e splicebert       unique_lm -0.003 [-0.005,-0.001] -0.006 [-0.012,-0.001] -0.002 [-0.004..+0.000] n=5                           -                           -      -0.003 [-0.005,-0.001] -0.005 [-0.006..-0.004] n=3
potential_at_c1prime_kT_e splicebert          shared +0.417 [+0.336,+0.527] +0.512 [+0.382,+0.693] +0.494 [+0.442..+0.562] n=5                           -                           -      +0.417 [+0.336,+0.527] +0.511 [+0.509..+0.512] n=3
potential_at_c1prime_kT_e splicebert unique_controls +0.379 [+0.305,+0.471] +0.284 [+0.155,+0.416] +0.345 [+0.268..+0.379] n=5                           -                           -      +0.379 [+0.305,+0.471] +0.285 [+0.283..+0.286] n=3
potential_at_c1prime_kT_e splicebert     unexplained +0.208 [+0.146,+0.272] +0.211 [+0.148,+0.278] +0.164 [+0.122..+0.187] n=5                           -                           -      +0.208 [+0.146,+0.272] +0.210 [+0.209..+0.211] n=3
  potential_mean_kT_e_12A   ernierna       unique_lm +0.015 [+0.002,+0.037] +0.010 [-0.002,+0.035] -0.003 [-0.010..+0.005] n=5 -0.004 [-0.010..+0.005] n=5 -0.004 [-0.007..-0.003] n=5 +0.017 [+0.000..+0.041] n=9 +0.009 [+0.001..+0.015] n=3
  potential_mean_kT_e_12A   ernierna          shared +0.448 [+0.279,+0.543] +0.069 [-0.022,+0.152] +0.121 [+0.032..+0.213] n=5 +0.121 [+0.032..+0.211] n=5 -0.011 [-0.046..+0.016] n=5 +0.446 [+0.428..+0.463] n=9 +0.069 [+0.064..+0.078] n=3
  potential_mean_kT_e_12A   ernierna unique_controls +0.296 [-0.039,+0.490] +0.674 [+0.342,+0.823] +0.679 [+0.562..+0.763] n=5 +0.679 [+0.563..+0.763] n=5 +0.811 [+0.762..+0.835] n=5 +0.286 [+0.277..+0.303] n=9 +0.674 [+0.666..+0.680] n=3
  potential_mean_kT_e_12A   ernierna     unexplained +0.241 [+0.107,+0.533] +0.247 [+0.111,+0.535] +0.204 [+0.182..+0.227] n=5 +0.204 [+0.182..+0.230] n=5 +0.205 [+0.181..+0.229] n=5 +0.251 [+0.237..+0.270] n=9 +0.247 [+0.242..+0.256] n=3
  potential_mean_kT_e_12A    rnabert       unique_lm +0.001 [-0.004,+0.008] +0.005 [-0.002,+0.021] -0.005 [-0.010..+0.001] n=5                           -                           - +0.012 [-0.006..+0.058] n=9 +0.005 [+0.003..+0.006] n=3
  potential_mean_kT_e_12A    rnabert          shared +0.111 [+0.020,+0.189] +0.077 [+0.013,+0.135] +0.011 [-0.011..+0.038] n=5                           -                           - +0.115 [+0.105..+0.122] n=9 +0.079 [+0.076..+0.082] n=3
  potential_mean_kT_e_12A    rnabert unique_controls +0.672 [+0.496,+0.814] +0.705 [+0.551,+0.834] +0.791 [+0.754..+0.827] n=5                           -                           - +0.641 [+0.587..+0.672] n=9 +0.704 [+0.701..+0.707] n=3
  potential_mean_kT_e_12A    rnabert     unexplained +0.216 [+0.096,+0.387] +0.212 [+0.095,+0.375] +0.203 [+0.187..+0.220] n=5                           -                           - +0.233 [+0.216..+0.286] n=9 +0.213 [+0.211..+0.214] n=3
  potential_mean_kT_e_12A      rnafm       unique_lm -0.002 [-0.008,+0.011] +0.004 [-0.001,+0.010] -0.001 [-0.008..+0.003] n=5 -0.001 [-0.007..+0.006] n=5 +0.003 [-0.001..+0.004] n=5 +0.009 [-0.018..+0.048] n=9 +0.009 [+0.002..+0.015] n=3
  potential_mean_kT_e_12A      rnafm          shared +0.409 [+0.202,+0.510] +0.074 [+0.020,+0.175] +0.083 [+0.022..+0.142] n=5 +0.103 [+0.022..+0.179] n=5 -0.014 [-0.088..+0.030] n=5 +0.401 [+0.366..+0.431] n=9 +0.069 [+0.063..+0.076] n=3
  potential_mean_kT_e_12A      rnafm unique_controls +0.358 [+0.052,+0.533] +0.692 [+0.372,+0.842] +0.710 [+0.642..+0.767] n=5 +0.689 [+0.611..+0.767] n=5 +0.806 [+0.761..+0.859] n=5 +0.339 [+0.329..+0.358] n=9 +0.698 [+0.691..+0.704] n=3
  potential_mean_kT_e_12A      rnafm     unexplained +0.235 [+0.113,+0.501] +0.229 [+0.108,+0.488] +0.209 [+0.183..+0.237] n=5 +0.209 [+0.178..+0.235] n=5 +0.205 [+0.185..+0.225] n=5 +0.251 [+0.235..+0.270] n=9 +0.225 [+0.218..+0.231] n=3
  potential_mean_kT_e_12A splicebert       unique_lm +0.014 [-0.003,+0.037] +0.005 [-0.007,+0.030] -0.005 [-0.010..+0.003] n=5                           -                           - +0.011 [+0.001..+0.028] n=9 +0.004 [-0.003..+0.011] n=3
  potential_mean_kT_e_12A splicebert          shared +0.206 [+0.069,+0.360] +0.074 [+0.000,+0.182] +0.041 [-0.008..+0.111] n=5                           -                           - +0.217 [+0.197..+0.234] n=9 +0.074 [+0.065..+0.083] n=3
  potential_mean_kT_e_12A splicebert unique_controls +0.538 [+0.141,+0.721] +0.670 [+0.304,+0.834] +0.758 [+0.664..+0.796] n=5                           -                           - +0.520 [+0.493..+0.543] n=9 +0.670 [+0.661..+0.679] n=3
  potential_mean_kT_e_12A splicebert     unexplained +0.242 [+0.105,+0.544] +0.251 [+0.112,+0.558] +0.205 [+0.179..+0.223] n=5                           -                           - +0.251 [+0.239..+0.280] n=9 +0.252 [+0.245..+0.259] n=3
   potential_mean_kT_e_4A   ernierna       unique_lm +0.001 [-0.004,+0.003] -0.006 [-0.016,+0.001] -0.004 [-0.010..+0.000] n=5 -0.003 [-0.010..+0.005] n=5 -0.005 [-0.010..-0.001] n=5 +0.012 [+0.001..+0.025] n=9 -0.001 [-0.005..+0.004] n=3
   potential_mean_kT_e_4A   ernierna          shared +0.402 [+0.275,+0.469] +0.076 [+0.027,+0.140] +0.158 [+0.095..+0.204] n=5 +0.156 [+0.095..+0.195] n=5 +0.003 [-0.029..+0.026] n=5 +0.389 [+0.378..+0.402] n=9 +0.064 [+0.058..+0.068] n=3
   potential_mean_kT_e_4A   ernierna unique_controls +0.345 [+0.189,+0.467] +0.671 [+0.502,+0.794] +0.615 [+0.585..+0.666] n=5 +0.617 [+0.591..+0.666] n=5 +0.770 [+0.740..+0.795] n=5 +0.347 [+0.331..+0.363] n=9 +0.683 [+0.679..+0.690] n=3
   potential_mean_kT_e_4A   ernierna     unexplained +0.252 [+0.140,+0.408] +0.259 [+0.142,+0.423] +0.231 [+0.210..+0.241] n=5 +0.230 [+0.206..+0.241] n=5 +0.232 [+0.221..+0.241] n=5 +0.252 [+0.237..+0.272] n=9 +0.253 [+0.248..+0.258] n=3
   potential_mean_kT_e_4A    rnabert       unique_lm -0.001 [-0.005,+0.004] -0.002 [-0.007,+0.002] -0.005 [-0.008..-0.002] n=5                           -                           - +0.006 [-0.006..+0.036] n=9 -0.005 [-0.010..-0.001] n=3
   potential_mean_kT_e_4A    rnabert          shared +0.138 [+0.075,+0.195] +0.117 [+0.075,+0.158] +0.070 [+0.044..+0.089] n=5                           -                           - +0.142 [+0.134..+0.148] n=9 +0.121 [+0.117..+0.126] n=3
   potential_mean_kT_e_4A    rnabert unique_controls +0.629 [+0.544,+0.726] +0.650 [+0.566,+0.743] +0.705 [+0.685..+0.732] n=5                           -                           - +0.600 [+0.561..+0.629] n=9 +0.646 [+0.640..+0.650] n=3
   potential_mean_kT_e_4A    rnabert     unexplained +0.234 [+0.156,+0.328] +0.235 [+0.154,+0.338] +0.229 [+0.211..+0.244] n=5                           -                           - +0.252 [+0.234..+0.288] n=9 +0.238 [+0.234..+0.244] n=3
   potential_mean_kT_e_4A      rnafm       unique_lm -0.003 [-0.016,+0.008] +0.009 [-0.002,+0.033] -0.004 [-0.008..+0.002] n=5 -0.003 [-0.010..+0.002] n=5 -0.005 [-0.012..+0.006] n=5 +0.002 [-0.019..+0.035] n=9 +0.001 [-0.002..+0.004] n=3
   potential_mean_kT_e_4A      rnafm          shared +0.351 [+0.190,+0.441] +0.085 [+0.027,+0.163] +0.126 [+0.091..+0.180] n=5 +0.137 [+0.091..+0.197] n=5 +0.059 [+0.042..+0.076] n=5 +0.343 [+0.317..+0.363] n=9 +0.092 [+0.089..+0.096] n=3
   potential_mean_kT_e_4A      rnafm unique_controls +0.404 [+0.234,+0.511] +0.670 [+0.499,+0.796] +0.644 [+0.584..+0.669] n=5 +0.632 [+0.567..+0.669] n=5 +0.711 [+0.688..+0.741] n=5 +0.394 [+0.382..+0.407] n=9 +0.663 [+0.659..+0.666] n=3
   potential_mean_kT_e_4A      rnafm     unexplained +0.248 [+0.135,+0.405] +0.236 [+0.139,+0.371] +0.234 [+0.211..+0.255] n=5 +0.234 [+0.211..+0.258] n=5 +0.235 [+0.218..+0.260] n=5 +0.261 [+0.248..+0.282] n=9 +0.244 [+0.241..+0.247] n=3
   potential_mean_kT_e_4A splicebert       unique_lm +0.004 [+0.001,+0.008] -0.001 [-0.009,+0.003] -0.002 [-0.004..-0.000] n=5                           -                           - +0.009 [-0.001..+0.024] n=9 +0.000 [-0.006..+0.004] n=3
   potential_mean_kT_e_4A splicebert          shared +0.199 [+0.109,+0.286] +0.093 [+0.053,+0.160] +0.079 [+0.045..+0.133] n=5                           -                           - +0.204 [+0.188..+0.221] n=9 +0.091 [+0.087..+0.099] n=3
   potential_mean_kT_e_4A splicebert unique_controls +0.548 [+0.359,+0.678] +0.654 [+0.479,+0.791] +0.694 [+0.656..+0.717] n=5                           -                           - +0.531 [+0.496..+0.556] n=9 +0.656 [+0.648..+0.661] n=3
   potential_mean_kT_e_4A splicebert     unexplained +0.249 [+0.141,+0.399] +0.253 [+0.141,+0.411] +0.229 [+0.211..+0.243] n=5                           -                           - +0.256 [+0.242..+0.278] n=9 +0.252 [+0.249..+0.258] n=3
```


### 4b. Variance partition: controls=controls_geom_coulomb, macro  (single file: value [95% bootstrap CI]; family folds: mean [min..max])

```
                   target      model       component             pretrained                 random                      family         matched_family_t0.5      matched_famrandom_t0.5          matched_pretrained              matched_random
potential_at_c1prime_kT_e   ernierna       unique_lm -0.000 [-0.002,+0.001] -0.000 [-0.001,+0.001] -0.003 [-0.004..-0.002] n=5 -0.003 [-0.005..-0.002] n=5 -0.001 [-0.003..+0.000] n=5      -0.000 [-0.002,+0.001] -0.001 [-0.001..-0.001] n=3
potential_at_c1prime_kT_e   ernierna          shared +0.667 [+0.612,+0.718] +0.694 [+0.649,+0.732] +0.699 [+0.670..+0.726] n=5 +0.698 [+0.670..+0.726] n=5 +0.521 [+0.487..+0.563] n=5      +0.667 [+0.612,+0.718] +0.580 [+0.565..+0.593] n=3
potential_at_c1prime_kT_e   ernierna unique_controls +0.211 [+0.167,+0.265] +0.184 [+0.157,+0.213] +0.192 [+0.154..+0.219] n=5 +0.192 [+0.154..+0.219] n=5 +0.370 [+0.338..+0.411] n=5      +0.211 [+0.167,+0.265] +0.298 [+0.285..+0.313] n=3
potential_at_c1prime_kT_e   ernierna     unexplained +0.122 [+0.106,+0.144] +0.122 [+0.106,+0.144] +0.112 [+0.101..+0.123] n=5 +0.113 [+0.101..+0.125] n=5 +0.111 [+0.099..+0.122] n=5      +0.122 [+0.106,+0.144] +0.123 [+0.123..+0.123] n=3
potential_at_c1prime_kT_e    rnabert       unique_lm -0.002 [-0.003,-0.001] -0.002 [-0.004,+0.000] -0.002 [-0.003..-0.001] n=5                           -                           -      -0.002 [-0.003,-0.001] -0.002 [-0.003..-0.002] n=3
potential_at_c1prime_kT_e    rnabert          shared +0.763 [+0.730,+0.791] +0.795 [+0.759,+0.822] +0.758 [+0.713..+0.784] n=5                           -                           -      +0.763 [+0.730,+0.791] +0.798 [+0.797..+0.800] n=3
potential_at_c1prime_kT_e    rnabert unique_controls +0.121 [+0.103,+0.140] +0.089 [+0.071,+0.110] +0.136 [+0.114..+0.167] n=5                           -                           -      +0.121 [+0.103,+0.140] +0.086 [+0.084..+0.087] n=3
potential_at_c1prime_kT_e    rnabert     unexplained +0.118 [+0.102,+0.139] +0.118 [+0.101,+0.140] +0.109 [+0.096..+0.123] n=5                           -                           -      +0.118 [+0.102,+0.139] +0.118 [+0.118..+0.119] n=3
potential_at_c1prime_kT_e      rnafm       unique_lm -0.002 [-0.004,-0.000] -0.001 [-0.003,+0.000] -0.002 [-0.003..-0.001] n=5 -0.002 [-0.003..-0.001] n=5 -0.001 [-0.002..-0.000] n=5      -0.002 [-0.004,-0.000] -0.001 [-0.002..+0.000] n=3
potential_at_c1prime_kT_e      rnafm          shared +0.747 [+0.710,+0.779] +0.785 [+0.748,+0.815] +0.722 [+0.604..+0.801] n=5 +0.738 [+0.669..+0.773] n=5 +0.780 [+0.712..+0.814] n=5      +0.747 [+0.710,+0.779] +0.788 [+0.784..+0.794] n=3
potential_at_c1prime_kT_e      rnafm unique_controls +0.131 [+0.113,+0.153] +0.093 [+0.074,+0.116] +0.169 [+0.100..+0.274] n=5 +0.154 [+0.129..+0.209] n=5 +0.111 [+0.088..+0.166] n=5      +0.131 [+0.113,+0.153] +0.091 [+0.085..+0.094] n=3
potential_at_c1prime_kT_e      rnafm     unexplained +0.124 [+0.107,+0.145] +0.123 [+0.106,+0.145] +0.110 [+0.099..+0.124] n=5 +0.110 [+0.099..+0.124] n=5 +0.110 [+0.098..+0.124] n=5      +0.124 [+0.107,+0.145] +0.123 [+0.121..+0.124] n=3
potential_at_c1prime_kT_e splicebert       unique_lm -0.001 [-0.002,-0.000] -0.000 [-0.002,+0.002] -0.002 [-0.003..-0.001] n=5                           -                           -      -0.001 [-0.002,-0.000] +0.000 [-0.001..+0.001] n=3
potential_at_c1prime_kT_e splicebert          shared +0.605 [+0.573,+0.635] +0.796 [+0.758,+0.826] +0.617 [+0.579..+0.714] n=5                           -                           -      +0.605 [+0.573,+0.635] +0.793 [+0.791..+0.795] n=3
potential_at_c1prime_kT_e splicebert unique_controls +0.272 [+0.251,+0.296] +0.082 [+0.062,+0.106] +0.273 [+0.166..+0.319] n=5                           -                           -      +0.272 [+0.251,+0.296] +0.084 [+0.082..+0.087] n=3
potential_at_c1prime_kT_e splicebert     unexplained +0.123 [+0.107,+0.145] +0.122 [+0.106,+0.144] +0.112 [+0.100..+0.123] n=5                           -                           -      +0.123 [+0.107,+0.145] +0.122 [+0.121..+0.123] n=3
  potential_mean_kT_e_12A   ernierna       unique_lm +0.003 [-0.005,+0.011] +0.005 [+0.000,+0.012] +0.001 [-0.005..+0.007] n=5 +0.001 [-0.005..+0.007] n=5 +0.000 [-0.005..+0.004] n=5 +0.029 [+0.003..+0.118] n=9 +0.005 [+0.003..+0.007] n=3
  potential_mean_kT_e_12A   ernierna          shared +0.166 [+0.035,+0.262] +0.076 [+0.040,+0.108] +0.065 [+0.023..+0.146] n=5 +0.068 [+0.023..+0.159] n=5 +0.058 [+0.012..+0.106] n=5 +0.146 [+0.084..+0.170] n=9 +0.081 [+0.079..+0.082] n=3
  potential_mean_kT_e_12A   ernierna unique_controls +0.600 [+0.490,+0.740] +0.690 [+0.600,+0.765] +0.741 [+0.626..+0.788] n=5 +0.738 [+0.613..+0.788] n=5 +0.748 [+0.718..+0.790] n=5 +0.556 [+0.465..+0.600] n=9 +0.685 [+0.683..+0.687] n=3
  potential_mean_kT_e_12A   ernierna     unexplained +0.231 [+0.162,+0.310] +0.229 [+0.160,+0.310] +0.193 [+0.166..+0.227] n=5 +0.193 [+0.166..+0.225] n=5 +0.194 [+0.169..+0.224] n=5 +0.270 [+0.231..+0.365] n=9 +0.229 [+0.227..+0.232] n=3
  potential_mean_kT_e_12A    rnabert       unique_lm -0.004 [-0.007,-0.001] +0.001 [-0.003,+0.008] -0.003 [-0.004..-0.002] n=5                           -                           - +0.030 [-0.004..+0.140] n=9 +0.001 [+0.001..+0.002] n=3
  potential_mean_kT_e_12A    rnabert          shared +0.063 [+0.008,+0.103] +0.072 [+0.034,+0.103] +0.026 [-0.008..+0.054] n=5                           -                           - +0.066 [+0.061..+0.074] n=9 +0.072 [+0.070..+0.075] n=3
  potential_mean_kT_e_12A    rnabert unique_controls +0.730 [+0.657,+0.805] +0.720 [+0.656,+0.786] +0.786 [+0.725..+0.847] n=5                           -                           - +0.657 [+0.539..+0.730] n=9 +0.721 [+0.718..+0.723] n=3
  potential_mean_kT_e_12A    rnabert     unexplained +0.211 [+0.151,+0.276] +0.206 [+0.150,+0.266] +0.192 [+0.165..+0.224] n=5                           -                           - +0.248 [+0.211..+0.353] n=9 +0.206 [+0.206..+0.207] n=3
  potential_mean_kT_e_12A      rnafm       unique_lm -0.003 [-0.007,+0.003] +0.001 [-0.003,+0.005] -0.000 [-0.005..+0.003] n=5 -0.000 [-0.003..+0.003] n=5 +0.001 [+0.000..+0.003] n=5 +0.025 [-0.005..+0.124] n=9 +0.003 [+0.001..+0.005] n=3
  potential_mean_kT_e_12A      rnafm          shared +0.181 [+0.065,+0.270] +0.048 [+0.010,+0.085] +0.083 [+0.036..+0.159] n=5 +0.094 [+0.046..+0.158] n=5 +0.007 [-0.066..+0.052] n=5 +0.166 [+0.114..+0.183] n=9 +0.043 [+0.039..+0.048] n=3
  potential_mean_kT_e_12A      rnafm unique_controls +0.594 [+0.499,+0.712] +0.727 [+0.633,+0.805] +0.722 [+0.618..+0.760] n=5 +0.711 [+0.619..+0.754] n=5 +0.798 [+0.743..+0.861] n=5 +0.539 [+0.467..+0.594] n=9 +0.732 [+0.727..+0.736] n=3
  potential_mean_kT_e_12A      rnafm     unexplained +0.228 [+0.163,+0.301] +0.224 [+0.160,+0.298] +0.196 [+0.166..+0.222] n=5 +0.195 [+0.166..+0.222] n=5 +0.194 [+0.168..+0.222] n=5 +0.270 [+0.228..+0.353] n=9 +0.222 [+0.220..+0.224] n=3
  potential_mean_kT_e_12A splicebert       unique_lm +0.006 [+0.000,+0.014] +0.001 [-0.003,+0.008] -0.002 [-0.005..+0.003] n=5                           -                           - +0.024 [+0.002..+0.103] n=9 +0.002 [-0.001..+0.006] n=3
  potential_mean_kT_e_12A splicebert          shared +0.087 [+0.013,+0.137] +0.055 [+0.019,+0.089] +0.058 [+0.007..+0.141] n=5                           -                           - +0.101 [+0.087..+0.113] n=9 +0.055 [+0.047..+0.059] n=3
  potential_mean_kT_e_12A splicebert unique_controls +0.679 [+0.583,+0.766] +0.710 [+0.608,+0.793] +0.748 [+0.631..+0.794] n=5                           -                           - +0.610 [+0.518..+0.679] n=9 +0.711 [+0.706..+0.718] n=3
  potential_mean_kT_e_12A splicebert     unexplained +0.228 [+0.158,+0.309] +0.233 [+0.163,+0.313] +0.196 [+0.174..+0.225] n=5                           -                           - +0.265 [+0.228..+0.354] n=9 +0.233 [+0.229..+0.235] n=3
   potential_mean_kT_e_4A   ernierna       unique_lm -0.006 [-0.013,+0.000] -0.005 [-0.009,-0.001] -0.003 [-0.008..+0.000] n=5 -0.004 [-0.008..+0.000] n=5 -0.004 [-0.005..-0.002] n=5 +0.012 [-0.006..+0.063] n=9 -0.001 [-0.001..+0.001] n=3
   potential_mean_kT_e_4A   ernierna          shared +0.200 [+0.150,+0.248] +0.111 [+0.087,+0.135] +0.143 [+0.117..+0.192] n=5 +0.144 [+0.117..+0.196] n=5 +0.050 [+0.017..+0.080] n=5 +0.179 [+0.137..+0.200] n=9 +0.077 [+0.072..+0.081] n=3
   potential_mean_kT_e_4A   ernierna unique_controls +0.518 [+0.471,+0.566] +0.606 [+0.556,+0.659] +0.615 [+0.588..+0.641] n=5 +0.614 [+0.588..+0.641] n=5 +0.708 [+0.660..+0.736] n=5 +0.499 [+0.430..+0.527] n=9 +0.640 [+0.637..+0.646] n=3
   potential_mean_kT_e_4A   ernierna     unexplained +0.289 [+0.251,+0.333] +0.287 [+0.248,+0.331] +0.245 [+0.217..+0.260] n=5 +0.245 [+0.221..+0.260] n=5 +0.245 [+0.221..+0.262] n=5 +0.310 [+0.289..+0.373] n=9 +0.283 [+0.282..+0.284] n=3
   potential_mean_kT_e_4A    rnabert       unique_lm -0.000 [-0.004,+0.003] -0.000 [-0.004,+0.003] -0.003 [-0.006..-0.001] n=5                           -                           - +0.015 [-0.003..+0.086] n=9 -0.001 [-0.002..+0.002] n=3
   potential_mean_kT_e_4A    rnabert          shared +0.111 [+0.085,+0.136] +0.126 [+0.102,+0.149] +0.082 [+0.054..+0.099] n=5                           -                           - +0.118 [+0.111..+0.131] n=9 +0.127 [+0.122..+0.129] n=3
   potential_mean_kT_e_4A    rnabert unique_controls +0.620 [+0.584,+0.660] +0.605 [+0.566,+0.643] +0.682 [+0.653..+0.711] n=5                           -                           - +0.571 [+0.478..+0.620] n=9 +0.605 [+0.602..+0.609] n=3
   potential_mean_kT_e_4A    rnabert     unexplained +0.269 [+0.239,+0.305] +0.269 [+0.238,+0.306] +0.239 [+0.218..+0.253] n=5                           -                           - +0.296 [+0.269..+0.378] n=9 +0.270 [+0.267..+0.271] n=3
   potential_mean_kT_e_4A      rnafm       unique_lm -0.003 [-0.008,+0.001] +0.002 [-0.002,+0.007] -0.002 [-0.005..+0.000] n=5 -0.003 [-0.005..-0.000] n=5 -0.002 [-0.005..+0.001] n=5 +0.011 [-0.009..+0.072] n=9 +0.000 [-0.002..+0.002] n=3
   potential_mean_kT_e_4A      rnafm          shared +0.181 [+0.128,+0.231] +0.106 [+0.082,+0.130] +0.139 [+0.099..+0.180] n=5 +0.145 [+0.117..+0.181] n=5 +0.084 [+0.064..+0.120] n=5 +0.178 [+0.157..+0.191] n=9 +0.101 [+0.097..+0.107] n=3
   potential_mean_kT_e_4A      rnafm unique_controls +0.540 [+0.491,+0.584] +0.615 [+0.568,+0.666] +0.617 [+0.570..+0.643] n=5 +0.611 [+0.566..+0.646] n=5 +0.671 [+0.627..+0.704] n=5 +0.502 [+0.447..+0.540] n=9 +0.620 [+0.614..+0.624] n=3
   potential_mean_kT_e_4A      rnafm     unexplained +0.282 [+0.242,+0.323] +0.276 [+0.241,+0.316] +0.246 [+0.217..+0.270] n=5 +0.247 [+0.217..+0.271] n=5 +0.246 [+0.221..+0.270] n=5 +0.309 [+0.282..+0.364] n=9 +0.278 [+0.277..+0.281] n=3
   potential_mean_kT_e_4A splicebert       unique_lm +0.003 [-0.000,+0.006] +0.001 [-0.005,+0.007] -0.002 [-0.004..+0.002] n=5                           -                           - +0.011 [-0.003..+0.052] n=9 +0.001 [+0.000..+0.002] n=3
   potential_mean_kT_e_4A splicebert          shared +0.125 [+0.090,+0.154] +0.113 [+0.088,+0.137] +0.103 [+0.062..+0.189] n=5                           -                           - +0.134 [+0.125..+0.146] n=9 +0.114 [+0.111..+0.119] n=3
   potential_mean_kT_e_4A splicebert unique_controls +0.593 [+0.545,+0.636] +0.604 [+0.555,+0.661] +0.655 [+0.595..+0.690] n=5                           -                           - +0.548 [+0.485..+0.593] n=9 +0.603 [+0.598..+0.606] n=3
   potential_mean_kT_e_4A splicebert     unexplained +0.280 [+0.243,+0.322] +0.281 [+0.242,+0.325] +0.244 [+0.219..+0.258] n=5                           -                           - +0.307 [+0.280..+0.365] n=9 +0.281 [+0.281..+0.282] n=3
```


### 4c. Predictor ladder, pretrained, macro R^2 (within-chain centered)

```
                   target      model  controls_geom  controls_geom+lm  controls_geom_coulomb  controls_geom_coulomb+lm  lm_gbm_pca  lm_ridge_full
potential_at_c1prime_kT_e   ernierna          0.839             0.837                  0.878                     0.878       0.667          0.586
potential_at_c1prime_kT_e    rnabert          0.847             0.846                  0.884                     0.882       0.761          0.706
potential_at_c1prime_kT_e      rnafm          0.841             0.839                  0.878                     0.876       0.745          0.712
potential_at_c1prime_kT_e splicebert          0.839             0.838                  0.878                     0.877       0.605          0.675
  potential_mean_kT_e_12A   ernierna          0.585             0.595                  0.766                     0.769       0.169         -0.116
  potential_mean_kT_e_12A    rnabert          0.604             0.603                  0.793                     0.789       0.059         -0.002
  potential_mean_kT_e_12A      rnafm          0.587             0.592                  0.775                     0.772       0.179         -0.176
  potential_mean_kT_e_12A splicebert          0.585             0.582                  0.766                     0.772       0.093         -0.268
   potential_mean_kT_e_4A   ernierna          0.529             0.525                  0.717                     0.711       0.193          0.065
   potential_mean_kT_e_4A    rnabert          0.531             0.535                  0.731                     0.731       0.110          0.077
   potential_mean_kT_e_4A      rnafm          0.531             0.529                  0.721                     0.718       0.178          0.006
   potential_mean_kT_e_4A splicebert          0.529             0.529                  0.717                     0.720       0.128         -0.011
```


### 4c. Predictor ladder, pretrained, micro R^2 (within-chain centered)

```
                   target      model  controls_geom  controls_geom+lm  controls_geom_coulomb  controls_geom_coulomb+lm  lm_gbm_pca  lm_ridge_full
potential_at_c1prime_kT_e   ernierna          0.706             0.721                  0.796                     0.798       0.605          0.528
potential_at_c1prime_kT_e    rnabert          0.766             0.764                  0.838                     0.836       0.627          0.565
potential_at_c1prime_kT_e      rnafm          0.708             0.715                  0.792                     0.789       0.556          0.513
potential_at_c1prime_kT_e splicebert          0.706             0.707                  0.796                     0.792       0.414          0.464
  potential_mean_kT_e_12A   ernierna          0.580             0.638                  0.744                     0.759       0.463          0.311
  potential_mean_kT_e_12A    rnabert          0.610             0.608                  0.783                     0.784       0.112          0.015
  potential_mean_kT_e_12A      rnafm          0.579             0.626                  0.767                     0.765       0.407          0.217
  potential_mean_kT_e_12A splicebert          0.580             0.594                  0.744                     0.758       0.220          0.105
   potential_mean_kT_e_4A   ernierna          0.586             0.623                  0.747                     0.748       0.403          0.267
   potential_mean_kT_e_4A    rnabert          0.581             0.583                  0.767                     0.766       0.137          0.059
   potential_mean_kT_e_4A      rnafm          0.587             0.606                  0.755                     0.752       0.348          0.189
   potential_mean_kT_e_4A splicebert          0.586             0.593                  0.747                     0.751       0.203          0.112
```


(random-init marker: /data/rna_electrostats/controls/random_init/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls/random_init/rnabert/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls/random_init/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls/random_init/splicebert/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold2_s0/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold2_s0/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold0_s0/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold0_s0/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/random_s0/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/random_s0/rnabert/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/random_s0/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/random_s0/splicebert/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold4_s0/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold4_s0/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold3_s0/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold3_s0/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/random_s2/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 2)


(random-init marker: /data/rna_electrostats/controls_matched/random_s2/rnabert/RANDOM_INIT.txt: random-initialised weights, seed 2)


(random-init marker: /data/rna_electrostats/controls_matched/random_s2/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 2)


(random-init marker: /data/rna_electrostats/controls_matched/random_s2/splicebert/RANDOM_INIT.txt: random-initialised weights, seed 2)


(random-init marker: /data/rna_electrostats/controls_matched/random_s1/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 1)


(random-init marker: /data/rna_electrostats/controls_matched/random_s1/rnabert/RANDOM_INIT.txt: random-initialised weights, seed 1)


(random-init marker: /data/rna_electrostats/controls_matched/random_s1/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 1)


(random-init marker: /data/rna_electrostats/controls_matched/random_s1/splicebert/RANDOM_INIT.txt: random-initialised weights, seed 1)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold1_s0/ernierna/RANDOM_INIT.txt: random-initialised weights, seed 0)


(random-init marker: /data/rna_electrostats/controls_matched/famrand_t0.5_fold1_s0/rnafm/RANDOM_INIT.txt: random-initialised weights, seed 0)


### 4d. Family split summary (first of 40 files: /data/rna_electrostats/controls/family/ernierna/fold0)

```
   family  n_residues  fold  n_chains  is_test_family
kmer_0000       71469     0       264            True
kmer_0001        9655     1        10           False
kmer_0014        4160     2         8           False
kmer_0013        3944     3         8           False
kmer_0007        3350     4        10           False
kmer_0025        2982     4         6           False
kmer_0022        2046     3         2           False
kmer_0006        1932     2        12           False
kmer_0039        1662     3         4           False
kmer_0003        1351     2        16           False
kmer_0032        1192     4         2           False
kmer_0004        1173     2        13           False
kmer_0168        1084     4         2           False
kmer_0184        1082     3         2           False
kmer_2475        1023     3         1           False
... (3061 more rows)
```


### 4e. Family split info (all folds)

```
                                         folder  fold  n_families  n_test_families  n_structures_dropped  n_structures_test  residues_test  residues_dropped family_leakage
                 controls/family/ernierna/fold0     0        3076                1                    71                112          15935             82185             []
                 controls/family/ernierna/fold1     1        3076              190                   472                197          11872             79612             []
                 controls/family/ernierna/fold2     2        3076              214                   452                229          20590             63392             []
                 controls/family/ernierna/fold3     3        3076              207                   444                209          19329             56081             []
                 controls/family/ernierna/fold4     4        3076              202                   460                202          20107             70244             []
                  controls/family/rnabert/fold0     0        3076                1                    71                112          13591             44685             []
                  controls/family/rnabert/fold1     1        3076              193                   468                210          13071             54582             []
                  controls/family/rnabert/fold2     2        3076              214                   454                211          16261             51773             []
                  controls/family/rnabert/fold3     3        3076              206                   437                209          18530             40712             []
                  controls/family/rnabert/fold4     4        3076              196                   466                203          18027             52356             []
                    controls/family/rnafm/fold0     0        3076                1                    71                112          15927             82059             []
                    controls/family/rnafm/fold1     1        3076              196                   464                204          12802             72793             []
                    controls/family/rnafm/fold2     2        3076              215                   447                233          19824             61150             []
                    controls/family/rnafm/fold3     3        3076              200                   460                194          17032             70517             []
                    controls/family/rnafm/fold4     4        3076              204                   459                204          20833             69835             []
               controls/family/splicebert/fold0     0        3076                1                    71                112          15935             82185             []
               controls/family/splicebert/fold1     1        3076              190                   472                197          11872             79612             []
               controls/family/splicebert/fold2     2        3076              214                   452                229          20590             63392             []
               controls/family/splicebert/fold3     3        3076              207                   444                209          19329             56081             []
               controls/family/splicebert/fold4     4        3076              202                   460                202          20107             70244             []
    controls_matched/family_t0.5_fold0/ernierna     0        3076                1                    71                112          15935             82185             []
       controls_matched/family_t0.5_fold0/rnafm     0        3076                1                    71                112          15927             82059             []
    controls_matched/family_t0.5_fold1/ernierna     1        3076              190                   472                197          11872             79612             []
       controls_matched/family_t0.5_fold1/rnafm     1        3076              196                   464                204          12802             72793             []
    controls_matched/family_t0.5_fold2/ernierna     2        3076              214                   452                229          20590             63392             []
       controls_matched/family_t0.5_fold2/rnafm     2        3076              215                   447                233          19824             61150             []
    controls_matched/family_t0.5_fold3/ernierna     3        3076              207                   444                209          19329             56081             []
       controls_matched/family_t0.5_fold3/rnafm     3        3076              200                   460                194          17032             70517             []
    controls_matched/family_t0.5_fold4/ernierna     4        3076              202                   460                202          20107             70244             []
       controls_matched/family_t0.5_fold4/rnafm     4        3076              204                   459                204          20833             69835             []
controls_matched/famrand_t0.5_fold0_s0/ernierna     0        3076                1                    71                112          15935             82185             []
   controls_matched/famrand_t0.5_fold0_s0/rnafm     0        3076                1                    71                112          15927             82059             []
controls_matched/famrand_t0.5_fold1_s0/ernierna     1        3076              190                   472                197          11872             79612             []
   controls_matched/famrand_t0.5_fold1_s0/rnafm     1        3076              196                   464                204          12802             72793             []
controls_matched/famrand_t0.5_fold2_s0/ernierna     2        3076              214                   452                229          20590             63392             []
   controls_matched/famrand_t0.5_fold2_s0/rnafm     2        3076              215                   447                233          19824             61150             []
controls_matched/famrand_t0.5_fold3_s0/ernierna     3        3076              207                   444                209          19329             56081             []
   controls_matched/famrand_t0.5_fold3_s0/rnafm     3        3076              200                   460                194          17032             70517             []
controls_matched/famrand_t0.5_fold4_s0/ernierna     4        3076              202                   460                202          20107             70244             []
   controls_matched/famrand_t0.5_fold4_s0/rnafm     4        3076              204                   459                204          20833             69835             []
```


### 4f. Layer used per matched run (LAYER.txt)

```
controls_matched/family_t0.5_fold0/ernierna: layer 11
controls_matched/family_t0.5_fold0/rnafm: layer 11
controls_matched/family_t0.5_fold1/ernierna: layer 11
controls_matched/family_t0.5_fold1/rnafm: layer 11
controls_matched/family_t0.5_fold2/ernierna: layer 11
controls_matched/family_t0.5_fold2/rnafm: layer 11
controls_matched/family_t0.5_fold3/ernierna: layer 11
controls_matched/family_t0.5_fold3/rnafm: layer 11
controls_matched/family_t0.5_fold4/ernierna: layer 11
controls_matched/family_t0.5_fold4/rnafm: layer 11
controls_matched/famrand_t0.5_fold0_s0/ernierna: layer 11
controls_matched/famrand_t0.5_fold0_s0/rnafm: layer 11
controls_matched/famrand_t0.5_fold1_s0/ernierna: layer 11
controls_matched/famrand_t0.5_fold1_s0/rnafm: layer 11
controls_matched/famrand_t0.5_fold2_s0/ernierna: layer 11
controls_matched/famrand_t0.5_fold2_s0/rnafm: layer 11
controls_matched/famrand_t0.5_fold3_s0/ernierna: layer 11
controls_matched/famrand_t0.5_fold3_s0/rnafm: layer 11
controls_matched/famrand_t0.5_fold4_s0/ernierna: layer 11
controls_matched/famrand_t0.5_fold4_s0/rnafm: layer 11
controls_matched/pretrained/ernierna: layer 11
controls_matched/pretrained/rnabert: layer 3
controls_matched/pretrained/rnafm: layer 11
controls_matched/pretrained/splicebert: layer 2
controls_matched/random_s0/ernierna: layer 11
controls_matched/random_s0/rnabert: layer 3
controls_matched/random_s0/rnafm: layer 11
controls_matched/random_s0/splicebert: layer 2
controls_matched/random_s1/ernierna: layer 11
controls_matched/random_s1/rnabert: layer 3
controls_matched/random_s1/rnafm: layer 11
controls_matched/random_s1/splicebert: layer 2
controls_matched/random_s2/ernierna: layer 11
controls_matched/random_s2/rnabert: layer 3
controls_matched/random_s2/rnafm: layer 11
controls_matched/random_s2/splicebert: layer 2
```


### 4g. Paired pretrained-minus-random (audit_extras diff): controls_matched/analysis/paired_differences.csv

```
       tag                    target              controls weighting component  a_value  b_mean  b_min  b_max  n_b_runs  diff_a_minus_b     lo     hi  n_structures  n_chains
   rnabert potential_at_c1prime_kT_e controls_geom_coulomb     micro unique_lm   -0.002  -0.003 -0.004 -0.003         3           0.001 -0.000  0.002           181       420
   rnabert potential_at_c1prime_kT_e controls_geom_coulomb     micro    shared    0.629   0.634  0.634  0.635         3          -0.006 -0.014  0.002           181       420
   rnabert potential_at_c1prime_kT_e controls_geom_coulomb     macro unique_lm   -0.002  -0.002 -0.003 -0.002         3           0.001 -0.001  0.002           181       420
   rnabert potential_at_c1prime_kT_e controls_geom_coulomb     macro    shared    0.763   0.798  0.797  0.800         3          -0.035 -0.047 -0.025           181       420
   rnabert potential_at_c1prime_kT_e         controls_geom     micro unique_lm   -0.002  -0.002 -0.003 -0.002         3           0.001 -0.001  0.002           181       420
   rnabert potential_at_c1prime_kT_e         controls_geom     micro    shared    0.628   0.634  0.633  0.634         3          -0.005 -0.014  0.002           181       420
   rnabert potential_at_c1prime_kT_e         controls_geom     macro unique_lm   -0.001  -0.001 -0.002 -0.001         3           0.000 -0.001  0.002           181       420
   rnabert potential_at_c1prime_kT_e         controls_geom     macro    shared    0.762   0.797  0.796  0.798         3          -0.035 -0.047 -0.025           181       420
   rnabert    potential_mean_kT_e_4A controls_geom_coulomb     micro unique_lm   -0.001  -0.005 -0.010 -0.001         3           0.004 -0.004  0.018           181       420
   rnabert    potential_mean_kT_e_4A controls_geom_coulomb     micro    shared    0.138   0.121  0.117  0.126         3           0.017 -0.011  0.043           181       420
   rnabert    potential_mean_kT_e_4A controls_geom_coulomb     macro unique_lm   -0.000  -0.001 -0.002  0.002         3           0.000 -0.003  0.005           181       420
   rnabert    potential_mean_kT_e_4A controls_geom_coulomb     macro    shared    0.111   0.127  0.122  0.129         3          -0.016 -0.032 -0.001           181       420
   rnabert    potential_mean_kT_e_4A         controls_geom     micro unique_lm    0.003  -0.002 -0.003 -0.002         3           0.005  0.000  0.011           181       420
   rnabert    potential_mean_kT_e_4A         controls_geom     micro    shared    0.135   0.118  0.117  0.119         3           0.017 -0.007  0.037           181       420
   rnabert    potential_mean_kT_e_4A         controls_geom     macro unique_lm    0.004  -0.004 -0.009  0.001         3           0.008 -0.000  0.021           181       420
   rnabert    potential_mean_kT_e_4A         controls_geom     macro    shared    0.106   0.130  0.126  0.136         3          -0.024 -0.043 -0.007           181       420
   rnabert   potential_mean_kT_e_12A controls_geom_coulomb     micro unique_lm    0.001   0.005  0.003  0.006         3          -0.004 -0.008 -0.000           181       420
   rnabert   potential_mean_kT_e_12A controls_geom_coulomb     micro    shared    0.111   0.079  0.076  0.082         3           0.032  0.003  0.058           181       420
   rnabert   potential_mean_kT_e_12A controls_geom_coulomb     macro unique_lm   -0.004   0.001  0.001  0.002         3          -0.005 -0.011 -0.000           181       420
   rnabert   potential_mean_kT_e_12A controls_geom_coulomb     macro    shared    0.063   0.072  0.070  0.075         3          -0.009 -0.033  0.008           181       420
   rnabert   potential_mean_kT_e_12A         controls_geom     micro unique_lm   -0.002  -0.002 -0.003  0.001         3          -0.000 -0.008  0.014           181       420
   rnabert   potential_mean_kT_e_12A         controls_geom     micro    shared    0.114   0.085  0.084  0.087         3           0.028 -0.005  0.057           181       420
   rnabert   potential_mean_kT_e_12A         controls_geom     macro unique_lm   -0.001  -0.000 -0.001  0.001         3          -0.001 -0.006  0.006           181       420
   rnabert   potential_mean_kT_e_12A         controls_geom     macro    shared    0.060   0.073  0.071  0.077         3          -0.014 -0.038  0.005           181       420
     rnafm potential_at_c1prime_kT_e controls_geom_coulomb     micro unique_lm   -0.003  -0.001 -0.003 -0.001         3          -0.002 -0.005  0.001           181       420
     rnafm potential_at_c1prime_kT_e controls_geom_coulomb     micro    shared    0.560   0.505  0.503  0.507         3           0.054 -0.025  0.119           181       420
     rnafm potential_at_c1prime_kT_e controls_geom_coulomb     macro unique_lm   -0.002  -0.001 -0.002  0.000         3          -0.001 -0.003  0.001           181       420
     rnafm potential_at_c1prime_kT_e controls_geom_coulomb     macro    shared    0.747   0.788  0.784  0.794         3          -0.041 -0.054 -0.025           181       420
     rnafm potential_at_c1prime_kT_e         controls_geom     micro unique_lm    0.008  -0.005 -0.006 -0.004         3           0.013 -0.001  0.023           181       420
     rnafm potential_at_c1prime_kT_e         controls_geom     micro    shared    0.549   0.509  0.507  0.510         3           0.040 -0.028  0.094           181       420
     rnafm potential_at_c1prime_kT_e         controls_geom     macro unique_lm   -0.002  -0.001 -0.003  0.000         3          -0.001 -0.004  0.003           181       420
     rnafm potential_at_c1prime_kT_e         controls_geom     macro    shared    0.747   0.788  0.783  0.795         3          -0.041 -0.054 -0.028           181       420
     rnafm    potential_mean_kT_e_4A controls_geom_coulomb     micro unique_lm   -0.003   0.001 -0.002  0.004         3          -0.004 -0.020  0.004           181       420
     rnafm    potential_mean_kT_e_4A controls_geom_coulomb     micro    shared    0.351   0.092  0.089  0.096         3           0.258  0.127  0.314           181       420
     rnafm    potential_mean_kT_e_4A controls_geom_coulomb     macro unique_lm   -0.003   0.000 -0.002  0.002         3          -0.003 -0.007 -0.000           181       420
     rnafm    potential_mean_kT_e_4A controls_geom_coulomb     macro    shared    0.181   0.101  0.097  0.107         3           0.079  0.026  0.139           181       420
     rnafm    potential_mean_kT_e_4A         controls_geom     micro unique_lm    0.020  -0.005 -0.008 -0.002         3           0.025  0.002  0.035           181       420
     rnafm    potential_mean_kT_e_4A         controls_geom     micro    shared    0.328   0.098  0.095  0.102         3           0.230  0.106  0.281           181       420
     rnafm    potential_mean_kT_e_4A         controls_geom     macro unique_lm   -0.002  -0.004 -0.006 -0.003         3           0.002 -0.009  0.011           181       420
     rnafm    potential_mean_kT_e_4A         controls_geom     macro    shared    0.180   0.106  0.102  0.111         3           0.074  0.027  0.126           181       420
     rnafm   potential_mean_kT_e_12A controls_geom_coulomb     micro unique_lm   -0.002   0.009  0.002  0.015         3          -0.011 -0.020  0.002           181       420
     rnafm   potential_mean_kT_e_12A controls_geom_coulomb     micro    shared    0.409   0.069  0.063  0.076         3           0.341  0.190  0.378           181       420
     rnafm   potential_mean_kT_e_12A controls_geom_coulomb     macro unique_lm   -0.003   0.003  0.001  0.005         3          -0.005 -0.009 -0.001           181       420
     rnafm   potential_mean_kT_e_12A controls_geom_coulomb     macro    shared    0.181   0.043  0.039  0.048         3           0.138  0.043  0.231           181       420
     rnafm   potential_mean_kT_e_12A         controls_geom     micro unique_lm    0.047  -0.008 -0.016 -0.002         3           0.055  0.019  0.064           181       420
     rnafm   potential_mean_kT_e_12A         controls_geom     micro    shared    0.360   0.085  0.078  0.094         3           0.275  0.143  0.309           181       420
     rnafm   potential_mean_kT_e_12A         controls_geom     macro unique_lm    0.006  -0.009 -0.013 -0.006         3           0.015 -0.008  0.033           181       420
     rnafm   potential_mean_kT_e_12A         controls_geom     macro    shared    0.173   0.055  0.051  0.058         3           0.118  0.040  0.191           181       420
splicebert potential_at_c1prime_kT_e controls_geom_coulomb     micro unique_lm   -0.003  -0.005 -0.006 -0.004         3           0.002 -0.000  0.004           181       420
splicebert potential_at_c1prime_kT_e controls_geom_coulomb     micro    shared    0.417   0.511  0.509  0.512         3          -0.094 -0.169 -0.038           181       420
splicebert potential_at_c1prime_kT_e controls_geom_coulomb     macro unique_lm   -0.001   0.000 -0.001  0.001         3          -0.001 -0.002  0.001           181       420
splicebert potential_at_c1prime_kT_e controls_geom_coulomb     macro    shared    0.605   0.793  0.791  0.795         3          -0.188 -0.206 -0.170           181       420
splicebert potential_at_c1prime_kT_e         controls_geom     micro unique_lm    0.001  -0.003 -0.003 -0.003         3           0.004 -0.000  0.007           181       420
splicebert potential_at_c1prime_kT_e         controls_geom     micro    shared    0.413   0.509  0.508  0.509         3          -0.096 -0.169 -0.041           181       420
splicebert potential_at_c1prime_kT_e         controls_geom     macro unique_lm   -0.000  -0.000 -0.002  0.002         3          -0.000 -0.002  0.002           181       420
splicebert potential_at_c1prime_kT_e         controls_geom     macro    shared    0.605   0.794  0.791  0.795         3          -0.189 -0.208 -0.171           181       420
splicebert    potential_mean_kT_e_4A controls_geom_coulomb     micro unique_lm    0.004   0.000 -0.006  0.004         3           0.004 -0.006  0.018           181       420
splicebert    potential_mean_kT_e_4A controls_geom_coulomb     micro    shared    0.199   0.091  0.087  0.099         3           0.108  0.046  0.133           181       420
splicebert    potential_mean_kT_e_4A controls_geom_coulomb     macro unique_lm    0.003   0.001  0.000  0.002         3           0.002 -0.004  0.008           181       420
splicebert    potential_mean_kT_e_4A controls_geom_coulomb     macro    shared    0.125   0.114  0.111  0.119         3           0.011 -0.019  0.040           181       420
splicebert    potential_mean_kT_e_4A         controls_geom     micro unique_lm    0.008  -0.004 -0.004 -0.003         3           0.011  0.006  0.018           181       420
splicebert    potential_mean_kT_e_4A         controls_geom     micro    shared    0.195   0.095  0.094  0.097         3           0.100  0.034  0.130           181       420
splicebert    potential_mean_kT_e_4A         controls_geom     macro unique_lm   -0.000  -0.006 -0.013  0.001         3           0.006 -0.003  0.016           181       420
splicebert    potential_mean_kT_e_4A         controls_geom     macro    shared    0.128   0.121  0.112  0.132         3           0.007 -0.026  0.037           181       420
splicebert   potential_mean_kT_e_12A controls_geom_coulomb     micro unique_lm    0.014   0.004 -0.003  0.011         3           0.010 -0.004  0.022           181       420
splicebert   potential_mean_kT_e_12A controls_geom_coulomb     micro    shared    0.206   0.074  0.065  0.083         3           0.132  0.051  0.196           181       420
splicebert   potential_mean_kT_e_12A controls_geom_coulomb     macro unique_lm    0.006   0.002 -0.001  0.006         3           0.005 -0.002  0.011           181       420
splicebert   potential_mean_kT_e_12A controls_geom_coulomb     macro    shared    0.087   0.055  0.047  0.059         3           0.032 -0.015  0.069           181       420
splicebert   potential_mean_kT_e_12A         controls_geom     micro unique_lm    0.014   0.000 -0.001  0.002         3           0.014 -0.003  0.019           181       420
splicebert   potential_mean_kT_e_12A         controls_geom     micro    shared    0.206   0.077  0.075  0.080         3           0.128  0.061  0.189           181       420
splicebert   potential_mean_kT_e_12A         controls_geom     macro unique_lm   -0.003  -0.000 -0.002  0.001         3          -0.002 -0.013  0.005           181       420
splicebert   potential_mean_kT_e_12A         controls_geom     macro    shared    0.096   0.057  0.055  0.059         3           0.039 -0.004  0.073           181       420
  ernierna potential_at_c1prime_kT_e controls_geom_coulomb     micro unique_lm    0.003  -0.004 -0.007  0.001         3           0.006  0.000  0.014           181       420
  ernierna potential_at_c1prime_kT_e controls_geom_coulomb     micro    shared    0.603   0.273  0.270  0.280         3           0.329  0.292  0.360           181       420
  ernierna potential_at_c1prime_kT_e controls_geom_coulomb     macro unique_lm   -0.000  -0.001 -0.001 -0.001         3           0.001 -0.001  0.002           181       420
  ernierna potential_at_c1prime_kT_e controls_geom_coulomb     macro    shared    0.667   0.580  0.565  0.593         3           0.087  0.028  0.144           181       420
  ernierna potential_at_c1prime_kT_e         controls_geom     micro unique_lm    0.015  -0.005 -0.006 -0.004         3           0.019  0.004  0.035           181       420
  ernierna potential_at_c1prime_kT_e         controls_geom     micro    shared    0.591   0.275  0.270  0.278         3           0.316  0.284  0.347           181       420
  ernierna potential_at_c1prime_kT_e         controls_geom     macro unique_lm   -0.001   0.001  0.000  0.002         3          -0.002 -0.005  0.001           181       420
  ernierna potential_at_c1prime_kT_e         controls_geom     macro    shared    0.668   0.578  0.564  0.592         3           0.090  0.031  0.146           181       420
  ernierna    potential_mean_kT_e_4A controls_geom_coulomb     micro unique_lm    0.001  -0.001 -0.005  0.004         3           0.002 -0.003  0.007           181       420
  ernierna    potential_mean_kT_e_4A controls_geom_coulomb     micro    shared    0.402   0.064  0.058  0.068         3           0.338  0.246  0.370           181       420
  ernierna    potential_mean_kT_e_4A controls_geom_coulomb     macro unique_lm   -0.006  -0.001 -0.001  0.001         3          -0.006 -0.011 -0.001           181       420
  ernierna    potential_mean_kT_e_4A controls_geom_coulomb     macro    shared    0.200   0.077  0.072  0.081         3           0.123  0.071  0.173           181       420
  ernierna    potential_mean_kT_e_4A         controls_geom     micro unique_lm    0.037  -0.011 -0.014 -0.008         3           0.048  0.022  0.063           181       420
  ernierna    potential_mean_kT_e_4A         controls_geom     micro    shared    0.366   0.075  0.073  0.076         3           0.292  0.217  0.325           181       420
  ernierna    potential_mean_kT_e_4A         controls_geom     macro unique_lm   -0.004  -0.005 -0.007 -0.004         3           0.001 -0.010  0.015           181       420
  ernierna    potential_mean_kT_e_4A         controls_geom     macro    shared    0.197   0.082  0.077  0.085         3           0.115  0.071  0.158           181       420
  ernierna   potential_mean_kT_e_12A controls_geom_coulomb     micro unique_lm    0.015   0.009  0.001  0.015         3           0.006 -0.006  0.021           181       420
  ernierna   potential_mean_kT_e_12A controls_geom_coulomb     micro    shared    0.448   0.069  0.064  0.078         3           0.379  0.284  0.432           181       420
  ernierna   potential_mean_kT_e_12A controls_geom_coulomb     macro unique_lm    0.003   0.005  0.003  0.007         3          -0.002 -0.010  0.005           181       420
  ernierna   potential_mean_kT_e_12A controls_geom_coulomb     macro    shared    0.166   0.081  0.079  0.082         3           0.085 -0.046  0.185           181       420
  ernierna   potential_mean_kT_e_12A         controls_geom     micro unique_lm    0.059  -0.015 -0.017 -0.014         3           0.074  0.032  0.094           181       420
  ernierna   potential_mean_kT_e_12A         controls_geom     micro    shared    0.405   0.094  0.092  0.096         3           0.311  0.241  0.368           181       420
  ernierna   potential_mean_kT_e_12A         controls_geom     macro unique_lm    0.009  -0.009 -0.010 -0.008         3           0.018 -0.009  0.041           181       420
  ernierna   potential_mean_kT_e_12A         controls_geom     macro    shared    0.159   0.095  0.091  0.097         3           0.065 -0.052  0.146           181       420
```


### 4h. Chain-length stratification (audit_extras length): controls_matched/analysis/length_partition.csv

```
  model        run                    target              controls weighting length_bin  n_chains  n_structures  n_residues component  value     lo     hi
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.002 -0.004 -0.001
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664    shared  0.629  0.545  0.718
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.002 -0.003 -0.001
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.763  0.732  0.790
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.001 -0.003 -0.000
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.573  0.449  0.654
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.004 -0.010  0.002
rnabert pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.283  0.231  0.385
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.001 -0.005  0.004
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664    shared  0.138  0.075  0.198
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.001 -0.004  0.003
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.100  0.065  0.138
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.002 -0.005  0.007
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.126  0.095  0.148
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.002 -0.011  0.008
rnabert pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.161  0.068  0.298
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664 unique_lm  0.001 -0.004  0.008
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664    shared  0.111  0.024  0.191
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.005 -0.008 -0.001
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.044 -0.016  0.097
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.001 -0.004  0.005
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.074  0.021  0.117
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm  0.002 -0.003  0.014
rnabert pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.137  0.045  0.315
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.003 -0.005 -0.001
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664    shared  0.634  0.543  0.727
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.002 -0.004 -0.000
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.776  0.741  0.804
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.001 -0.001  0.003
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.563  0.434  0.651
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.007 -0.013 -0.002
rnabert  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.275  0.220  0.384
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.004 -0.009  0.000
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664    shared  0.119  0.081  0.159
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.001 -0.006  0.003
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.100  0.068  0.136
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.003 -0.002  0.008
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.120  0.089  0.141
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.007 -0.019 -0.000
rnabert  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.129  0.071  0.218
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664 unique_lm  0.005 -0.001  0.012
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664    shared  0.080  0.018  0.140
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.001 -0.004  0.001
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.048 -0.017  0.100
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.000 -0.003  0.003
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.078  0.033  0.106
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm  0.007 -0.001  0.019
rnabert  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.089  0.019  0.210
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.003 -0.004 -0.001
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664    shared  0.635  0.543  0.729
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.003 -0.004 -0.001
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.779  0.746  0.806
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.002 -0.005  0.000
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.565  0.432  0.654
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.004 -0.007  0.001
rnabert  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.271  0.217  0.372
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.010 -0.028  0.001
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664    shared  0.126  0.091  0.165
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm  0.000 -0.005  0.005
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.100  0.070  0.131
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.003 -0.009  0.003
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.120  0.077  0.145
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.018 -0.052  0.001
rnabert  random_s1    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.142  0.092  0.228
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664 unique_lm  0.003 -0.003  0.013
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664    shared  0.082  0.017  0.147
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.003 -0.006  0.001
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.046 -0.006  0.093
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.002 -0.007  0.002
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.079  0.028  0.119
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm  0.006 -0.002  0.020
rnabert  random_s1   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.093  0.022  0.225
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.004 -0.005 -0.002
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       16664    shared  0.635  0.542  0.732
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.003 -0.005 -0.002
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.780  0.746  0.808
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.002 -0.005  0.001
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.566  0.435  0.654
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.006 -0.012 -0.001
rnabert  random_s2 potential_at_c1prime_kT_e controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.267  0.213  0.382
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664 unique_lm -0.001 -0.006  0.003
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       16664    shared  0.117  0.076  0.160
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm  0.002 -0.003  0.007
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.099  0.066  0.139
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.004 -0.002  0.009
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.123  0.095  0.140
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm -0.004 -0.013  0.003
rnabert  random_s2    potential_mean_kT_e_4A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.124  0.063  0.215
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664 unique_lm  0.006 -0.003  0.023
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       16664    shared  0.076  0.007  0.142
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.001 -0.004  0.003
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.049 -0.008  0.097
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.006 -0.014  0.002
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.085  0.042  0.119
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370 unique_lm  0.011 -0.001  0.039
rnabert  random_s2   potential_mean_kT_e_12A controls_geom_coulomb     micro    251-500        10             7        4370    shared  0.082 -0.004  0.213
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       22004 unique_lm -0.003 -0.009  0.002
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       22004    shared  0.560  0.475  0.678
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.002 -0.004 -0.000
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.728  0.693  0.760
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.003 -0.005 -0.000
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.616  0.509  0.696
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro       >500        10             7        9710 unique_lm -0.004 -0.022  0.006
  rnafm pretrained potential_at_c1prime_kT_e controls_geom_coulomb     micro       >500        10             7        9710    shared  0.367  0.279  0.463
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       22004 unique_lm -0.003 -0.016  0.008
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       22004    shared  0.351  0.169  0.441
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.004 -0.009  0.000
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.152  0.113  0.194
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.002 -0.012  0.005
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.222  0.176  0.282
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro       >500        10             7        9710 unique_lm -0.003 -0.022  0.015
  rnafm pretrained    potential_mean_kT_e_4A controls_geom_coulomb     micro       >500        10             7        9710    shared  0.395  0.232  0.532
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       22004 unique_lm -0.002 -0.009  0.012
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       22004    shared  0.409  0.211  0.517
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.005 -0.012  0.003
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.079  0.040  0.121
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.002 -0.014  0.018
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.170  0.101  0.256
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro       >500        10             7        9710 unique_lm -0.002 -0.009  0.014
  rnafm pretrained   potential_mean_kT_e_12A controls_geom_coulomb     micro       >500        10             7        9710    shared  0.447  0.321  0.617
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       22004 unique_lm -0.003 -0.007  0.002
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       22004    shared  0.507  0.375  0.682
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm  0.001 -0.001  0.002
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.771  0.739  0.798
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.002 -0.002  0.007
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.557  0.418  0.651
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro       >500        10             7        9710 unique_lm -0.007 -0.017  0.003
  rnafm  random_s0 potential_at_c1prime_kT_e controls_geom_coulomb     micro       >500        10             7        9710    shared  0.215  0.152  0.332
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       22004 unique_lm  0.004 -0.012  0.034
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro        all       420           181       22004    shared  0.089  0.026  0.174
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm  0.000 -0.006  0.006
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.090  0.063  0.120
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.005 -0.002  0.010
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.102  0.077  0.118
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro       >500        10             7        9710 unique_lm  0.005 -0.015  0.045
  rnafm  random_s0    potential_mean_kT_e_4A controls_geom_coulomb     micro       >500        10             7        9710    shared  0.088  0.014  0.205
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       22004 unique_lm  0.009  0.002  0.022
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro        all       420           181       22004    shared  0.067  0.009  0.163
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm  0.001 -0.002  0.005
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.022 -0.024  0.057
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm  0.007 -0.004  0.028
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.057  0.017  0.085
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro       >500        10             7        9710 unique_lm  0.010  0.003  0.028
  rnafm  random_s0   potential_mean_kT_e_12A controls_geom_coulomb     micro       >500        10             7        9710    shared  0.071  0.026  0.186
  rnafm  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       22004 unique_lm -0.001 -0.005  0.003
  rnafm  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro        all       420           181       22004    shared  0.506  0.371  0.687
  rnafm  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237 unique_lm -0.002 -0.004  0.000
  rnafm  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro      1-100       386           168        9237    shared  0.777  0.743  0.803
  rnafm  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057 unique_lm -0.000 -0.003  0.002
  rnafm  random_s1 potential_at_c1prime_kT_e controls_geom_coulomb     micro    101-250        24            17        3057    shared  0.559  0.425  0.649
... (234 more rows)
```


### 4i. Spike-in power analysis (recovered unique_lm): controls_matched/pretrained/ernierna/spike_in_summary.csv

```
                 target weighting  value     lo    hi  injected  spike_type  control_explained_raw_signal
potential_mean_kT_e_12A     micro  0.008  0.000 0.026     0.005      linear                         0.749
potential_mean_kT_e_12A     macro  0.006 -0.005 0.015     0.005      linear                         0.749
potential_mean_kT_e_12A     micro  0.000 -0.021 0.014     0.010      linear                         0.749
potential_mean_kT_e_12A     macro  0.014  0.004 0.028     0.010      linear                         0.749
potential_mean_kT_e_12A     micro  0.018 -0.001 0.050     0.020      linear                         0.749
potential_mean_kT_e_12A     macro  0.042  0.024 0.072     0.020      linear                         0.749
potential_mean_kT_e_12A     micro  0.041  0.023 0.091     0.050      linear                         0.749
potential_mean_kT_e_12A     macro  0.118  0.075 0.169     0.050      linear                         0.749
 potential_mean_kT_e_4A     micro  0.011 -0.007 0.037     0.005      linear                         0.749
 potential_mean_kT_e_4A     macro -0.001 -0.006 0.007     0.005      linear                         0.749
 potential_mean_kT_e_4A     micro  0.005 -0.001 0.011     0.010      linear                         0.749
 potential_mean_kT_e_4A     macro  0.011  0.005 0.018     0.010      linear                         0.749
 potential_mean_kT_e_4A     micro  0.019  0.002 0.043     0.020      linear                         0.749
 potential_mean_kT_e_4A     macro  0.019  0.009 0.028     0.020      linear                         0.749
 potential_mean_kT_e_4A     micro  0.025  0.007 0.057     0.050      linear                         0.749
 potential_mean_kT_e_4A     macro  0.063  0.045 0.080     0.050      linear                         0.749
potential_mean_kT_e_12A     micro  0.018  0.006 0.055     0.005 interaction                         0.168
potential_mean_kT_e_12A     macro  0.010  0.004 0.019     0.005 interaction                         0.168
potential_mean_kT_e_12A     micro  0.028  0.009 0.074     0.010 interaction                         0.168
potential_mean_kT_e_12A     macro  0.014  0.005 0.030     0.010 interaction                         0.168
potential_mean_kT_e_12A     micro  0.004 -0.009 0.012     0.020 interaction                         0.168
potential_mean_kT_e_12A     macro  0.013  0.007 0.020     0.020 interaction                         0.168
potential_mean_kT_e_12A     micro  0.017  0.006 0.052     0.050 interaction                         0.168
potential_mean_kT_e_12A     macro  0.038  0.024 0.059     0.050 interaction                         0.168
 potential_mean_kT_e_4A     micro  0.021  0.010 0.035     0.005 interaction                         0.168
 potential_mean_kT_e_4A     macro  0.004 -0.003 0.009     0.005 interaction                         0.168
 potential_mean_kT_e_4A     micro  0.004 -0.009 0.020     0.010 interaction                         0.168
 potential_mean_kT_e_4A     macro -0.001 -0.008 0.005     0.010 interaction                         0.168
 potential_mean_kT_e_4A     micro  0.006 -0.001 0.012     0.020 interaction                         0.168
 potential_mean_kT_e_4A     macro  0.001 -0.002 0.005     0.020 interaction                         0.168
 potential_mean_kT_e_4A     micro  0.016  0.011 0.027     0.050 interaction                         0.168
 potential_mean_kT_e_4A     macro  0.017  0.013 0.022     0.050 interaction                         0.168
```


### 4i. Spike-in power analysis (recovered unique_lm): controls_matched/pretrained/rnabert/spike_in_summary.csv

```
                 target weighting  value     lo    hi  injected  spike_type  control_explained_raw_signal
potential_mean_kT_e_12A     micro  0.014 -0.002 0.040     0.005      linear                         0.658
potential_mean_kT_e_12A     macro  0.012  0.002 0.022     0.005      linear                         0.658
potential_mean_kT_e_12A     micro  0.015  0.001 0.033     0.010      linear                         0.658
potential_mean_kT_e_12A     macro  0.027  0.013 0.044     0.010      linear                         0.658
potential_mean_kT_e_12A     micro  0.016  0.006 0.030     0.020      linear                         0.658
potential_mean_kT_e_12A     macro  0.052  0.032 0.089     0.020      linear                         0.658
potential_mean_kT_e_12A     micro  0.058  0.033 0.092     0.050      linear                         0.658
potential_mean_kT_e_12A     macro  0.140  0.096 0.214     0.050      linear                         0.658
 potential_mean_kT_e_4A     micro -0.006 -0.015 0.002     0.005      linear                         0.658
 potential_mean_kT_e_4A     macro  0.001 -0.003 0.008     0.005      linear                         0.658
 potential_mean_kT_e_4A     micro -0.000 -0.005 0.005     0.010      linear                         0.658
 potential_mean_kT_e_4A     macro  0.009  0.003 0.017     0.010      linear                         0.658
 potential_mean_kT_e_4A     micro  0.013  0.003 0.026     0.020      linear                         0.658
 potential_mean_kT_e_4A     macro  0.027  0.015 0.040     0.020      linear                         0.658
 potential_mean_kT_e_4A     micro  0.036  0.025 0.051     0.050      linear                         0.658
 potential_mean_kT_e_4A     macro  0.086  0.063 0.119     0.050      linear                         0.658
potential_mean_kT_e_12A     micro -0.006 -0.017 0.002     0.005 interaction                         0.267
potential_mean_kT_e_12A     macro -0.003 -0.008 0.001     0.005 interaction                         0.267
potential_mean_kT_e_12A     micro -0.000 -0.007 0.008     0.010 interaction                         0.267
potential_mean_kT_e_12A     macro  0.002 -0.003 0.008     0.010 interaction                         0.267
potential_mean_kT_e_12A     micro -0.003 -0.014 0.005     0.020 interaction                         0.267
potential_mean_kT_e_12A     macro  0.005 -0.004 0.014     0.020 interaction                         0.267
potential_mean_kT_e_12A     micro  0.009  0.002 0.028     0.050 interaction                         0.267
potential_mean_kT_e_12A     macro  0.040  0.026 0.063     0.050 interaction                         0.267
 potential_mean_kT_e_4A     micro -0.005 -0.014 0.001     0.005 interaction                         0.267
 potential_mean_kT_e_4A     macro -0.003 -0.006 0.001     0.005 interaction                         0.267
 potential_mean_kT_e_4A     micro  0.003 -0.003 0.012     0.010 interaction                         0.267
 potential_mean_kT_e_4A     macro  0.001 -0.004 0.007     0.010 interaction                         0.267
 potential_mean_kT_e_4A     micro  0.001 -0.008 0.011     0.020 interaction                         0.267
 potential_mean_kT_e_4A     macro -0.002 -0.007 0.002     0.020 interaction                         0.267
 potential_mean_kT_e_4A     micro  0.009  0.002 0.014     0.050 interaction                         0.267
 potential_mean_kT_e_4A     macro  0.012  0.007 0.018     0.050 interaction                         0.267
```


### 4i. Spike-in power analysis (recovered unique_lm): controls_matched/pretrained/rnafm/spike_in_summary.csv

```
                 target weighting  value     lo     hi  injected  spike_type  control_explained_raw_signal
potential_mean_kT_e_12A     micro  0.011 -0.012  0.035     0.005      linear                         0.725
potential_mean_kT_e_12A     macro  0.008 -0.005  0.020     0.005      linear                         0.725
potential_mean_kT_e_12A     micro  0.003 -0.001  0.013     0.010      linear                         0.725
potential_mean_kT_e_12A     macro  0.019  0.008  0.031     0.010      linear                         0.725
potential_mean_kT_e_12A     micro  0.015 -0.001  0.049     0.020      linear                         0.725
potential_mean_kT_e_12A     macro  0.044  0.022  0.070     0.020      linear                         0.725
potential_mean_kT_e_12A     micro  0.048  0.020  0.102     0.050      linear                         0.725
potential_mean_kT_e_12A     macro  0.124  0.085  0.171     0.050      linear                         0.725
 potential_mean_kT_e_4A     micro -0.010 -0.015  0.001     0.005      linear                         0.725
 potential_mean_kT_e_4A     macro  0.001 -0.003  0.006     0.005      linear                         0.725
 potential_mean_kT_e_4A     micro  0.007 -0.014  0.029     0.010      linear                         0.725
 potential_mean_kT_e_4A     macro  0.007  0.001  0.014     0.010      linear                         0.725
 potential_mean_kT_e_4A     micro -0.000 -0.006  0.010     0.020      linear                         0.725
 potential_mean_kT_e_4A     macro  0.023  0.013  0.031     0.020      linear                         0.725
 potential_mean_kT_e_4A     micro  0.035  0.026  0.050     0.050      linear                         0.725
 potential_mean_kT_e_4A     macro  0.072  0.055  0.087     0.050      linear                         0.725
potential_mean_kT_e_12A     micro -0.018 -0.056  0.006     0.005 interaction                         0.293
potential_mean_kT_e_12A     macro -0.005 -0.015  0.003     0.005 interaction                         0.293
potential_mean_kT_e_12A     micro -0.008 -0.022  0.005     0.010 interaction                         0.293
potential_mean_kT_e_12A     macro  0.003 -0.001  0.010     0.010 interaction                         0.293
potential_mean_kT_e_12A     micro  0.014 -0.000  0.030     0.020 interaction                         0.293
potential_mean_kT_e_12A     macro  0.011  0.004  0.017     0.020 interaction                         0.293
potential_mean_kT_e_12A     micro  0.013  0.005  0.030     0.050 interaction                         0.293
potential_mean_kT_e_12A     macro  0.026  0.013  0.040     0.050 interaction                         0.293
 potential_mean_kT_e_4A     micro -0.019 -0.046 -0.004     0.005 interaction                         0.293
 potential_mean_kT_e_4A     macro -0.009 -0.016 -0.003     0.005 interaction                         0.293
 potential_mean_kT_e_4A     micro  0.004 -0.001  0.012     0.010 interaction                         0.293
 potential_mean_kT_e_4A     macro -0.002 -0.005  0.001     0.010 interaction                         0.293
 potential_mean_kT_e_4A     micro -0.003 -0.015  0.013     0.020 interaction                         0.293
 potential_mean_kT_e_4A     macro  0.001 -0.004  0.004     0.020 interaction                         0.293
 potential_mean_kT_e_4A     micro  0.007  0.001  0.018     0.050 interaction                         0.293
 potential_mean_kT_e_4A     macro  0.013  0.008  0.019     0.050 interaction                         0.293
```


### 4i. Spike-in power analysis (recovered unique_lm): controls_matched/pretrained/splicebert/spike_in_summary.csv

```
                 target weighting  value     lo    hi  injected  spike_type  control_explained_raw_signal
potential_mean_kT_e_12A     micro  0.001 -0.006 0.006     0.005      linear                         0.690
potential_mean_kT_e_12A     macro  0.002 -0.009 0.010     0.005      linear                         0.690
potential_mean_kT_e_12A     micro  0.003 -0.015 0.034     0.010      linear                         0.690
potential_mean_kT_e_12A     macro  0.014  0.002 0.033     0.010      linear                         0.690
potential_mean_kT_e_12A     micro  0.011  0.001 0.040     0.020      linear                         0.690
potential_mean_kT_e_12A     macro  0.036  0.018 0.064     0.020      linear                         0.690
potential_mean_kT_e_12A     micro  0.028  0.009 0.070     0.050      linear                         0.690
potential_mean_kT_e_12A     macro  0.103  0.066 0.162     0.050      linear                         0.690
 potential_mean_kT_e_4A     micro -0.001 -0.012 0.009     0.005      linear                         0.690
 potential_mean_kT_e_4A     macro -0.003 -0.007 0.003     0.005      linear                         0.690
 potential_mean_kT_e_4A     micro  0.004 -0.018 0.026     0.010      linear                         0.690
 potential_mean_kT_e_4A     macro  0.005 -0.003 0.013     0.010      linear                         0.690
 potential_mean_kT_e_4A     micro  0.012  0.006 0.020     0.020      linear                         0.690
 potential_mean_kT_e_4A     macro  0.017  0.010 0.026     0.020      linear                         0.690
 potential_mean_kT_e_4A     micro  0.024  0.006 0.049     0.050      linear                         0.690
 potential_mean_kT_e_4A     macro  0.052  0.037 0.069     0.050      linear                         0.690
potential_mean_kT_e_12A     micro  0.021  0.003 0.060     0.005 interaction                         0.209
potential_mean_kT_e_12A     macro  0.005 -0.004 0.016     0.005 interaction                         0.209
potential_mean_kT_e_12A     micro  0.006 -0.017 0.017     0.010 interaction                         0.209
potential_mean_kT_e_12A     macro  0.004 -0.003 0.012     0.010 interaction                         0.209
potential_mean_kT_e_12A     micro  0.012  0.002 0.044     0.020 interaction                         0.209
potential_mean_kT_e_12A     macro  0.010  0.004 0.018     0.020 interaction                         0.209
potential_mean_kT_e_12A     micro  0.007 -0.011 0.031     0.050 interaction                         0.209
potential_mean_kT_e_12A     macro  0.035  0.022 0.049     0.050 interaction                         0.209
 potential_mean_kT_e_4A     micro  0.015  0.001 0.037     0.005 interaction                         0.209
 potential_mean_kT_e_4A     macro  0.000 -0.005 0.005     0.005 interaction                         0.209
 potential_mean_kT_e_4A     micro  0.009  0.003 0.015     0.010 interaction                         0.209
 potential_mean_kT_e_4A     macro  0.001 -0.003 0.005     0.010 interaction                         0.209
 potential_mean_kT_e_4A     micro  0.002 -0.008 0.017     0.020 interaction                         0.209
 potential_mean_kT_e_4A     macro  0.000 -0.005 0.005     0.020 interaction                         0.209
 potential_mean_kT_e_4A     micro  0.013  0.002 0.030     0.050 interaction                         0.209
 potential_mean_kT_e_4A     macro  0.019  0.013 0.026     0.050 interaction                         0.209
```


## 5. Physics baseline (coulomb_residual_report.csv)

```
target                     model  raw_test_r  centered_test_r  raw_test_r2  centered_test_r2
   c1p                  corr_vac       0.641            0.294          NaN               NaN
   c1p            corr_dhS_I0.15       0.657            0.472          NaN               NaN
   c1p            corr_dhP_I0.15       0.657            0.472          NaN               NaN
   c1p        calibrated_coulomb         NaN              NaN        0.465             0.231
   c1p         controls_only_gbm         NaN              NaN          NaN             0.308
   c1p          coulomb_only_gbm         NaN              NaN          NaN             0.305
   c1p controls_plus_coulomb_gbm         NaN              NaN          NaN             0.436
     p                  corr_vac       0.889            0.087          NaN               NaN
     p            corr_dhS_I0.15       0.613            0.138          NaN               NaN
     p            corr_dhP_I0.15       0.613            0.138          NaN               NaN
     p        calibrated_coulomb         NaN              NaN        0.756             0.020
     p         controls_only_gbm         NaN              NaN          NaN             0.013
     p          coulomb_only_gbm         NaN              NaN          NaN             0.026
     p controls_plus_coulomb_gbm         NaN              NaN          NaN             0.029
    m4                  corr_vac       0.450            0.403          NaN               NaN
    m4            corr_dhS_I0.15       0.664            0.663          NaN               NaN
    m4            corr_dhP_I0.15       0.664            0.663          NaN               NaN
    m4        calibrated_coulomb         NaN              NaN        0.441             0.463
    m4         controls_only_gbm         NaN              NaN          NaN             0.496
    m4          coulomb_only_gbm         NaN              NaN          NaN             0.597
    m4 controls_plus_coulomb_gbm         NaN              NaN          NaN             0.721
    m8                  corr_vac       0.523            0.475          NaN               NaN
    m8            corr_dhS_I0.15       0.705            0.699          NaN               NaN
    m8            corr_dhP_I0.15       0.705            0.699          NaN               NaN
    m8        calibrated_coulomb         NaN              NaN        0.487             0.498
    m8         controls_only_gbm         NaN              NaN          NaN             0.520
    m8          coulomb_only_gbm         NaN              NaN          NaN             0.641
    m8 controls_plus_coulomb_gbm         NaN              NaN          NaN             0.745
   m12                  corr_vac       0.569            0.547          NaN               NaN
   m12            corr_dhS_I0.15       0.729            0.732          NaN               NaN
   m12            corr_dhP_I0.15       0.729            0.732          NaN               NaN
   m12        calibrated_coulomb         NaN              NaN        0.517             0.533
   m12         controls_only_gbm         NaN              NaN          NaN             0.454
   m12          coulomb_only_gbm         NaN              NaN          NaN             0.691
   m12 controls_plus_coulomb_gbm         NaN              NaN          NaN             0.734
```


### 6. centered_latent/cross_model/cross_model_agreement_centered_base_resid.csv

```
               pair  nuisance_removed  raw_scores  shuffle_global  shuffle_within_chain
   rnabert-ernierna             0.144       0.169          -0.002                 0.120
      rnabert-rnafm             0.171       0.187          -0.002                 0.150
 rnabert-splicebert             0.224       0.264          -0.002                 0.186
     rnafm-ernierna             0.512       0.545          -0.001                 0.271
   rnafm-splicebert             0.299       0.429          -0.001                 0.159
splicebert-ernierna             0.268       0.357           0.005                 0.093
```


### 6. centered_latent_v2/cross_model/cross_model_agreement_centered_base_resid.csv  controls=False residualizer=linear

```
               pair  nuisance_removed  raw_scores  shuffle_global  shuffle_within_chain  shuffle_within_chain_v1  within_chain_clean
   rnabert-ernierna             0.064       0.191           0.008                 0.002                    0.038               0.028
      rnabert-rnafm             0.092       0.242           0.005                -0.000                    0.063               0.070
 rnabert-splicebert             0.149       0.224           0.001                 0.004                    0.108               0.083
     rnafm-ernierna             0.381       0.486          -0.004                 0.001                    0.196               0.300
   rnafm-splicebert             0.227       0.318          -0.003                -0.001                    0.105               0.190
splicebert-ernierna             0.256       0.314          -0.003                -0.000                    0.032               0.255
```


### 6. centered_latent_v2/cross_model_gbm/cross_model_agreement_centered_struct_gbm_ctrl_gbm.csv  controls=True residualizer=gbm

```
               pair  nuisance_removed  raw_scores  shuffle_global  shuffle_within_chain  shuffle_within_chain_v1  within_chain_clean
   rnabert-ernierna             0.071       0.171          -0.001                -0.000                    0.049               0.023
      rnabert-rnafm             0.126       0.229          -0.005                -0.001                    0.086               0.023
 rnabert-splicebert             0.134       0.120           0.004                -0.004                    0.112               0.021
     rnafm-ernierna             0.463       0.521          -0.005                 0.005                    0.206               0.352
   rnafm-splicebert             0.330       0.383          -0.001                 0.003                    0.115               0.266
splicebert-ernierna             0.316       0.405          -0.002                -0.003                    0.059               0.297
```


### 6. centered_latent_v3/cross_model/cross_model_agreement_centered_base_resid.csv  controls=none residualizer=linear

```
               pair  nuisance_removed  raw_scores  shuffle_global  shuffle_within_chain  shuffle_within_chain_v1  within_chain_clean
   rnabert-ernierna             0.064       0.191          -0.001                -0.000                    0.038               0.028
      rnabert-rnafm             0.092       0.242           0.002                -0.001                    0.063               0.070
 rnabert-splicebert             0.149       0.224          -0.000                 0.004                    0.109               0.083
     rnafm-ernierna             0.381       0.486           0.000                 0.004                    0.196               0.300
   rnafm-splicebert             0.227       0.318           0.001                 0.001                    0.103               0.190
splicebert-ernierna             0.256       0.314          -0.000                -0.000                    0.033               0.255
```


### 6. centered_latent_v3/cross_model/cross_model_agreement_centered_dssr_ctx_gbm_ctrl_struct-dssr-ctx_gbm.csv  controls=struct+dssr+ctx residualizer=gbm

```
               pair  nuisance_removed  raw_scores  shuffle_global  shuffle_within_chain  shuffle_within_chain_v1  within_chain_clean
   rnabert-ernierna             0.075       0.229          -0.001                 0.001                    0.059               0.016
      rnabert-rnafm             0.110       0.267           0.002                -0.004                    0.077               0.018
 rnabert-splicebert             0.138       0.167          -0.005                 0.000                    0.138               0.015
     rnafm-ernierna             0.444       0.512           0.003                -0.001                    0.269               0.203
   rnafm-splicebert             0.207       0.283           0.000                 0.004                    0.112               0.107
splicebert-ernierna             0.204       0.258           0.005                 0.001                    0.081               0.129
```


### 6. centered_latent_v3/cross_model/cross_model_agreement_swap_geom.csv  controls=none residualizer=linear

```
               pair  nuisance_removed  raw_scores  shuffle_global  shuffle_within_chain  shuffle_within_chain_v1  within_chain_clean
   rnabert-ernierna             0.057       0.140          -0.001                -0.003                    0.044               0.037
      rnabert-rnafm             0.122       0.177           0.008                -0.006                    0.103               0.037
 rnabert-splicebert             0.137       0.253           0.005                -0.002                    0.117               0.080
     rnafm-ernierna             0.488       0.562           0.003                -0.002                    0.194               0.397
   rnafm-splicebert             0.376       0.514          -0.003                 0.001                    0.152               0.279
splicebert-ernierna             0.258       0.353           0.001                -0.003                    0.066               0.259
```


## 7. Benchmark: bench_apbs_v2

```
file: /data/rna_electrostats/bench_apbs_v2/chain_metrics_with_excess.csv
chains ok: 189 (not ok: 0)
solver: {'nonlinear': 189}
median C1' RMSD 5.11  median lDDT 0.696
```


### bench_apbs_v2: summary.txt

```
chains scored: 189 of 189 (failed: 0)
median C1' RMSD 5.11 A, median C1'-lDDT 0.696

[c1p] pred-vs-native within-chain Pearson: median 0.611 (95% CI 0.572..0.638); median W1/sd 0.429
    vs RMSD-matched 'jitter' null (median 0.570): excess = pred - null median -0.052 (CI -0.155..+0.006); pred worse in 57% of chains; Wilcoxon p=0.000942
    vs RMSD-matched 'jitterL' null (median 0.835): excess = pred - null median -0.249 (CI -0.264..-0.209); pred worse in 87% of chains; Wilcoxon p=9.27e-25
    vs RMSD-matched 'smooth' null (median 0.389): excess = pred - null median +0.108 (CI +0.028..+0.162); pred worse in 39% of chains; Wilcoxon p=8.51e-06
    vs RMSD-matched 'smoothL' null (median 0.685): excess = pred - null median -0.127 (CI -0.151..-0.078); pred worse in 74% of chains; Wilcoxon p=1e-10
    Spearman(RMSD, pred Pearson) = -0.35; Spearman(lDDT, pred Pearson) = 0.46

[m4] pred-vs-native within-chain Pearson: median 0.406 (95% CI 0.316..0.486); median W1/sd 0.970
    vs RMSD-matched 'jitter' null (median 0.327): excess = pred - null median +0.018 (CI -0.018..+0.074); pred worse in 44% of chains; Wilcoxon p=0.0813
    vs RMSD-matched 'jitterL' null (median 0.687): excess = pred - null median -0.222 (CI -0.261..-0.194); pred worse in 96% of chains; Wilcoxon p=2.58e-29
    vs RMSD-matched 'smooth' null (median 0.298): excess = pred - null median +0.012 (CI -0.023..+0.083); pred worse in 48% of chains; Wilcoxon p=0.0878
    vs RMSD-matched 'smoothL' null (median 0.580): excess = pred - null median -0.156 (CI -0.189..-0.126); pred worse in 84% of chains; Wilcoxon p=3.05e-21
    Spearman(RMSD, pred Pearson) = -0.72; Spearman(lDDT, pred Pearson) = 0.71

[m8] pred-vs-native within-chain Pearson: median 0.600 (95% CI 0.466..0.716); median W1/sd 0.725
    vs RMSD-matched 'jitter' null (median 0.424): excess = pred - null median +0.091 (CI +0.049..+0.122); pred worse in 28% of chains; Wilcoxon p=5.43e-10
    vs RMSD-matched 'jitterL' null (median 0.821): excess = pred - null median -0.166 (CI -0.205..-0.122); pred worse in 96% of chains; Wilcoxon p=2.11e-30
    vs RMSD-matched 'smooth' null (median 0.438): excess = pred - null median +0.038 (CI +0.004..+0.088); pred worse in 40% of chains; Wilcoxon p=0.000551
    vs RMSD-matched 'smoothL' null (median 0.695): excess = pred - null median -0.086 (CI -0.113..-0.067); pred worse in 74% of chains; Wilcoxon p=2.68e-13
    Spearman(RMSD, pred Pearson) = -0.74; Spearman(lDDT, pred Pearson) = 0.76

[m12] pred-vs-native within-chain Pearson: median 0.758 (95% CI 0.593..0.834); median W1/sd 0.613
    vs RMSD-matched 'jitter' null (median 0.581): excess = pred - null median +0.054 (CI +0.022..+0.085); pred worse in 32% of chains; Wilcoxon p=8.75e-08
    vs RMSD-matched 'jitterL' null (median 0.919): excess = pred - null median -0.147 (CI -0.200..-0.107); pred worse in 98% of chains; Wilcoxon p=2.09e-32
    vs RMSD-matched 'smooth' null (median 0.555): excess = pred - null median +0.046 (CI +0.003..+0.079); pred worse in 41% of chains; Wilcoxon p=0.000135
    vs RMSD-matched 'smoothL' null (median 0.793): excess = pred - null median -0.058 (CI -0.087..-0.036); pred worse in 70% of chains; Wilcoxon p=1.95e-10
    Spearman(RMSD, pred Pearson) = -0.76; Spearman(lDDT, pred Pearson) = 0.80

Null kinds: smooth/jitter are matched on C1' RMSD; smoothL/jitterL are matched on C1' lDDT.

Interpretation guide:
  excess < 0 with CI below 0 : the electrostatic map is damaged MORE than geometry error of the same size would predict -> the field carries information RMSD does not.
  excess ~ 0                 : the map is just a function of RMSD; no added value over RMSD/lDDT.
  excess > 0                 : predictions are electrostatically better than RMSD suggests.
```


### bench_apbs_v2: excess_vs_null.csv

```
qty    null  median_excess  ci_lo  ci_hi  frac_pred_worse   n
c1p  jitter         -0.052 -0.155  0.006            0.566 189
c1p jitterL         -0.249 -0.264 -0.209            0.873 189
c1p  smooth          0.108  0.028  0.162            0.386 189
c1p smoothL         -0.127 -0.151 -0.078            0.741 189
 m4  jitter          0.018 -0.018  0.074            0.444 189
 m4 jitterL         -0.222 -0.261 -0.194            0.958 189
 m4  smooth          0.012 -0.023  0.083            0.476 189
 m4 smoothL         -0.156 -0.189 -0.126            0.841 189
 m8  jitter          0.091  0.049  0.122            0.275 189
 m8 jitterL         -0.166 -0.205 -0.122            0.963 189
 m8  smooth          0.038  0.004  0.088            0.402 189
 m8 smoothL         -0.086 -0.113 -0.067            0.741 189
m12  jitter          0.054  0.022  0.085            0.317 189
m12 jitterL         -0.147 -0.200 -0.107            0.979 189
m12  smooth          0.046  0.003  0.079            0.413 189
m12 smoothL         -0.058 -0.087 -0.036            0.698 189
```


### bench_apbs_v2: fidelity vs lDDT / length

```
Median within-chain Pearson (pred vs native) by C1'-lDDT bin
quantity      c1p   m12    m4    m8
lddt_bin                           
(0.0, 0.4]  0.277 0.244 0.186 0.298
(0.4, 0.5]  0.370 0.282 0.104 0.235
(0.5, 0.6]  0.508 0.395 0.213 0.213
(0.6, 0.7]  0.608 0.601 0.305 0.427
(0.7, 0.8]  0.641 0.855 0.535 0.717
(0.8, 1.01] 0.655 0.929 0.689 0.870

chains per lDDT bin:
lddt_bin
(0.0, 0.4]     22
(0.4, 0.5]     19
(0.5, 0.6]     17
(0.6, 0.7]     38
(0.7, 0.8]     45
(0.8, 1.01]    48

By chain length (median):
            pred_pearson_c1p  pred_pearson_m4  pred_pearson_m8  pred_pearson_m12  lddt_c1p  rmsd_c1p   n
len_bin                                                                                                 
(0, 50]                0.629            0.370            0.550             0.697     0.703     5.576  69
(50, 100]              0.602            0.404            0.649             0.742     0.700     4.836  86
(100, 150]             0.629            0.608            0.808             0.882     0.719     3.717  28
(150, 260]             0.542            0.438            0.601             0.770     0.599     8.982   6

Standardised OLS, Pearson ~ lDDT + log RMSD + log length:
quantity  b_lddt  b_log_rmsd  b_log_len    R2
     c1p   0.118       0.015     -0.023 0.276
      m4   0.052      -0.146      0.005 0.509
      m8   0.121      -0.120      0.005 0.551
     m12   0.162      -0.084     -0.000 0.581
```


## 7. Benchmark: bench_coulomb_v2

```
file: /data/rna_electrostats/bench_coulomb_v2/chain_metrics_with_excess.csv
chains ok: 189 (not ok: 0)
solver: {'coulomb_dhP_I0.15': 189}
median C1' RMSD 5.11  median lDDT 0.696
```


### bench_coulomb_v2: summary.txt

```
chains scored: 189 of 189 (failed: 0)
median C1' RMSD 5.11 A, median C1'-lDDT 0.696

[c1p] pred-vs-native within-chain Pearson: median 0.652 (95% CI 0.584..0.720); median W1/sd 0.497
    vs RMSD-matched 'jitter' null (median 0.489): excess = pred - null median +0.085 (CI +0.052..+0.125); pred worse in 35% of chains; Wilcoxon p=1.13e-06
    vs RMSD-matched 'jitterL' null (median 0.880): excess = pred - null median -0.161 (CI -0.207..-0.129); pred worse in 95% of chains; Wilcoxon p=1.5e-29
    vs RMSD-matched 'smooth' null (median 0.452): excess = pred - null median +0.103 (CI +0.056..+0.169); pred worse in 32% of chains; Wilcoxon p=9.56e-12
    vs RMSD-matched 'smoothL' null (median 0.739): excess = pred - null median -0.070 (CI -0.088..-0.047); pred worse in 76% of chains; Wilcoxon p=4.52e-11
    Spearman(RMSD, pred Pearson) = -0.73; Spearman(lDDT, pred Pearson) = 0.77

[m4] pred-vs-native within-chain Pearson: median 0.749 (95% CI 0.657..0.814); median W1/sd 0.336
    vs RMSD-matched 'jitter' null (median 0.646): excess = pred - null median +0.046 (CI +0.017..+0.068); pred worse in 35% of chains; Wilcoxon p=1.45e-07
    vs RMSD-matched 'jitterL' null (median 0.920): excess = pred - null median -0.146 (CI -0.203..-0.103); pred worse in 98% of chains; Wilcoxon p=1.69e-31
    vs RMSD-matched 'smooth' null (median 0.604): excess = pred - null median +0.069 (CI +0.031..+0.114); pred worse in 32% of chains; Wilcoxon p=1.32e-09
    vs RMSD-matched 'smoothL' null (median 0.834): excess = pred - null median -0.057 (CI -0.092..-0.039); pred worse in 77% of chains; Wilcoxon p=1.33e-15
    Spearman(RMSD, pred Pearson) = -0.79; Spearman(lDDT, pred Pearson) = 0.82

[m8] pred-vs-native within-chain Pearson: median 0.827 (95% CI 0.762..0.882); median W1/sd 0.245
    vs RMSD-matched 'jitter' null (median 0.733): excess = pred - null median +0.043 (CI +0.024..+0.066); pred worse in 25% of chains; Wilcoxon p=3.89e-13
    vs RMSD-matched 'jitterL' null (median 0.953): excess = pred - null median -0.118 (CI -0.155..-0.071); pred worse in 99% of chains; Wilcoxon p=2.69e-32
    vs RMSD-matched 'smooth' null (median 0.715): excess = pred - null median +0.051 (CI +0.023..+0.087); pred worse in 32% of chains; Wilcoxon p=6.32e-10
    vs RMSD-matched 'smoothL' null (median 0.899): excess = pred - null median -0.050 (CI -0.081..-0.026); pred worse in 81% of chains; Wilcoxon p=2.25e-18
    Spearman(RMSD, pred Pearson) = -0.85; Spearman(lDDT, pred Pearson) = 0.88

[m12] pred-vs-native within-chain Pearson: median 0.855 (95% CI 0.795..0.920); median W1/sd 0.198
    vs RMSD-matched 'jitter' null (median 0.794): excess = pred - null median +0.034 (CI +0.018..+0.056); pred worse in 24% of chains; Wilcoxon p=1.25e-12
    vs RMSD-matched 'jitterL' null (median 0.973): excess = pred - null median -0.110 (CI -0.142..-0.057); pred worse in 100% of chains; Wilcoxon p=9.14e-33
    vs RMSD-matched 'smooth' null (median 0.797): excess = pred - null median +0.030 (CI +0.017..+0.053); pred worse in 31% of chains; Wilcoxon p=3.15e-08
    vs RMSD-matched 'smoothL' null (median 0.933): excess = pred - null median -0.047 (CI -0.083..-0.021); pred worse in 84% of chains; Wilcoxon p=5.48e-22
    Spearman(RMSD, pred Pearson) = -0.86; Spearman(lDDT, pred Pearson) = 0.89

Null kinds: smooth/jitter are matched on C1' RMSD; smoothL/jitterL are matched on C1' lDDT.

Interpretation guide:
  excess < 0 with CI below 0 : the electrostatic map is damaged MORE than geometry error of the same size would predict -> the field carries information RMSD does not.
  excess ~ 0                 : the map is just a function of RMSD; no added value over RMSD/lDDT.
  excess > 0                 : predictions are electrostatically better than RMSD suggests.
```


### bench_coulomb_v2: excess_vs_null.csv

```
qty    null  median_excess  ci_lo  ci_hi  frac_pred_worse   n
c1p  jitter          0.085  0.052  0.125            0.349 189
c1p jitterL         -0.161 -0.207 -0.129            0.947 189
c1p  smooth          0.103  0.056  0.169            0.323 189
c1p smoothL         -0.070 -0.088 -0.047            0.757 189
 m4  jitter          0.046  0.017  0.068            0.354 189
 m4 jitterL         -0.146 -0.203 -0.103            0.984 189
 m4  smooth          0.069  0.031  0.114            0.317 189
 m4 smoothL         -0.057 -0.092 -0.039            0.772 189
 m8  jitter          0.043  0.024  0.066            0.254 189
 m8 jitterL         -0.118 -0.155 -0.071            0.989 189
 m8  smooth          0.051  0.023  0.087            0.323 189
 m8 smoothL         -0.050 -0.081 -0.026            0.810 189
m12  jitter          0.034  0.018  0.056            0.243 189
m12 jitterL         -0.110 -0.142 -0.057            1.000 189
m12  smooth          0.030  0.017  0.053            0.312 189
m12 smoothL         -0.047 -0.083 -0.021            0.836 189
```


### bench_coulomb_v2: fidelity vs lDDT / length

```
Median within-chain Pearson (pred vs native) by C1'-lDDT bin
quantity      c1p   m12    m4    m8
lddt_bin                           
(0.0, 0.4]  0.239 0.394 0.362 0.332
(0.4, 0.5]  0.401 0.451 0.394 0.460
(0.5, 0.6]  0.548 0.756 0.549 0.657
(0.6, 0.7]  0.639 0.796 0.647 0.771
(0.7, 0.8]  0.726 0.924 0.846 0.887
(0.8, 1.01] 0.853 0.980 0.921 0.965

chains per lDDT bin:
lddt_bin
(0.0, 0.4]     22
(0.4, 0.5]     19
(0.5, 0.6]     17
(0.6, 0.7]     38
(0.7, 0.8]     45
(0.8, 1.01]    48

By chain length (median):
            pred_pearson_c1p  pred_pearson_m4  pred_pearson_m8  pred_pearson_m12  lddt_c1p  rmsd_c1p   n
len_bin                                                                                                 
(0, 50]                0.632            0.701            0.790             0.833     0.703     5.576  69
(50, 100]              0.644            0.737            0.833             0.861     0.700     4.836  86
(100, 150]             0.812            0.904            0.945             0.956     0.719     3.717  28
(150, 260]             0.769            0.840            0.882             0.912     0.599     8.982   6

Standardised OLS, Pearson ~ lDDT + log RMSD + log length:
quantity  b_lddt  b_log_rmsd  b_log_len    R2
     c1p   0.145      -0.054      0.035 0.624
      m4   0.160      -0.049      0.014 0.674
      m8   0.176      -0.042      0.008 0.698
     m12   0.174      -0.045      0.018 0.677
```


## 7. Benchmark: bench_out

```
file: /data/rna_electrostats/bench_out/chain_metrics_with_excess.csv
chains ok: 189 (not ok: 0)
solver: {'coulomb_dhP_I0.15': 189}
median C1' RMSD 5.11  median lDDT 0.696
```


### bench_out: summary.txt

```
chains scored: 189 of 189 (failed: 0)
median C1' RMSD 5.11 A, median C1'-lDDT 0.696

[c1p] pred-vs-native within-chain Pearson: median 0.652 (95% CI 0.584..0.720); median W1/sd 0.497
    vs RMSD-matched 'jitter' null (median 0.489): excess = pred - null median +0.085 (CI +0.052..+0.125); pred worse in 35% of chains; Wilcoxon p=1.13e-06
    vs RMSD-matched 'smooth' null (median 0.452): excess = pred - null median +0.103 (CI +0.056..+0.169); pred worse in 32% of chains; Wilcoxon p=9.56e-12
    Spearman(RMSD, pred Pearson) = -0.73; Spearman(lDDT, pred Pearson) = 0.77

[m4] pred-vs-native within-chain Pearson: median 0.749 (95% CI 0.657..0.814); median W1/sd 0.336
    vs RMSD-matched 'jitter' null (median 0.646): excess = pred - null median +0.046 (CI +0.017..+0.068); pred worse in 35% of chains; Wilcoxon p=1.45e-07
    vs RMSD-matched 'smooth' null (median 0.604): excess = pred - null median +0.069 (CI +0.031..+0.114); pred worse in 32% of chains; Wilcoxon p=1.32e-09
    Spearman(RMSD, pred Pearson) = -0.79; Spearman(lDDT, pred Pearson) = 0.82

[m8] pred-vs-native within-chain Pearson: median 0.827 (95% CI 0.762..0.882); median W1/sd 0.245
    vs RMSD-matched 'jitter' null (median 0.733): excess = pred - null median +0.043 (CI +0.024..+0.066); pred worse in 25% of chains; Wilcoxon p=3.89e-13
    vs RMSD-matched 'smooth' null (median 0.715): excess = pred - null median +0.051 (CI +0.023..+0.087); pred worse in 32% of chains; Wilcoxon p=6.32e-10
    Spearman(RMSD, pred Pearson) = -0.85; Spearman(lDDT, pred Pearson) = 0.88

[m12] pred-vs-native within-chain Pearson: median 0.855 (95% CI 0.795..0.920); median W1/sd 0.198
    vs RMSD-matched 'jitter' null (median 0.794): excess = pred - null median +0.034 (CI +0.018..+0.056); pred worse in 24% of chains; Wilcoxon p=1.25e-12
    vs RMSD-matched 'smooth' null (median 0.797): excess = pred - null median +0.030 (CI +0.017..+0.053); pred worse in 31% of chains; Wilcoxon p=3.15e-08
    Spearman(RMSD, pred Pearson) = -0.86; Spearman(lDDT, pred Pearson) = 0.89

Null kinds: smooth/jitter are matched on C1' RMSD; smoothL/jitterL are matched on C1' lDDT.

Interpretation guide:
  excess < 0 with CI below 0 : the electrostatic map is damaged MORE than geometry error of the same size would predict -> the field carries information RMSD does not.
  excess ~ 0                 : the map is just a function of RMSD; no added value over RMSD/lDDT.
  excess > 0                 : predictions are electrostatically better than RMSD suggests.
```


### bench_out: excess_vs_null.csv

```
qty   null  median_excess  ci_lo  ci_hi  frac_pred_worse   n
c1p jitter          0.085  0.052  0.125            0.349 189
c1p smooth          0.103  0.056  0.169            0.323 189
 m4 jitter          0.046  0.017  0.068            0.354 189
 m4 smooth          0.069  0.031  0.114            0.317 189
 m8 jitter          0.043  0.024  0.066            0.254 189
 m8 smooth          0.051  0.023  0.087            0.323 189
m12 jitter          0.034  0.018  0.056            0.243 189
m12 smooth          0.030  0.017  0.053            0.312 189
```


### bench_out: fidelity vs lDDT / length

```
Median within-chain Pearson (pred vs native) by C1'-lDDT bin
quantity      c1p   m12    m4    m8
lddt_bin                           
(0.0, 0.4]  0.239 0.394 0.362 0.332
(0.4, 0.5]  0.401 0.451 0.394 0.460
(0.5, 0.6]  0.548 0.756 0.549 0.657
(0.6, 0.7]  0.639 0.796 0.647 0.771
(0.7, 0.8]  0.726 0.924 0.846 0.887
(0.8, 1.01] 0.853 0.980 0.921 0.965

chains per lDDT bin:
lddt_bin
(0.0, 0.4]     22
(0.4, 0.5]     19
(0.5, 0.6]     17
(0.6, 0.7]     38
(0.7, 0.8]     45
(0.8, 1.01]    48

By chain length (median):
            pred_pearson_c1p  pred_pearson_m4  pred_pearson_m8  pred_pearson_m12  lddt_c1p  rmsd_c1p   n
len_bin                                                                                                 
(0, 50]                0.632            0.701            0.790             0.833     0.703     5.576  69
(50, 100]              0.644            0.737            0.833             0.861     0.700     4.836  86
(100, 150]             0.812            0.904            0.945             0.956     0.719     3.717  28
(150, 260]             0.769            0.840            0.882             0.912     0.599     8.982   6

Standardised OLS, Pearson ~ lDDT + log RMSD + log length:
quantity  b_lddt  b_log_rmsd  b_log_len    R2
     c1p   0.145      -0.054      0.035 0.624
      m4   0.160      -0.049      0.014 0.674
      m8   0.176      -0.042      0.008 0.698
     m12   0.174      -0.045      0.018 0.677
```


## 8. mg_full

```
labels_r0.79/all_residue_potentials.csv  2MB
labels_r0.79/work/1J1U/1j1u.fasta  83B
labels_r0.79/work/1J1U/ions.csv  587B
labels_r0.79/work/1J1U/per_residue_potential.csv  19KB
labels_r0.79/work/1J1U/residue_renumber_map.csv  1KB
labels_r0.79/work/1J1U/seq_index_map.csv  2KB
labels_r0.79/work/1J1U/structure_metadata.csv  153B
labels_r0.79/work/1L2X/1l2x.fasta  36B
labels_r0.79/work/1L2X/ions.csv  1KB
labels_r0.79/work/1L2X/per_residue_potential.csv  7KB
labels_r0.79/work/1L2X/residue_renumber_map.csv  553B
labels_r0.79/work/1L2X/seq_index_map.csv  712B
labels_r0.79/work/1L2X/structure_metadata.csv  153B
labels_r0.79/work/1L9A/1l9a.fasta  134B
labels_r0.79/work/1L9A/ions.csv  2KB
labels_r0.79/work/1L9A/per_residue_potential.csv  32KB
labels_r0.79/work/1L9A/residue_renumber_map.csv  2KB
labels_r0.79/work/1L9A/seq_index_map.csv  3KB
labels_r0.79/work/1L9A/structure_metadata.csv  152B
labels_r0.79/work/1LNT/1lnt.fasta  40B
labels_r0.79/work/1LNT/ions.csv  826B
labels_r0.79/work/1LNT/per_residue_potential.csv  6KB
labels_r0.79/work/1LNT/residue_renumber_map.csv  453B
labels_r0.79/work/1LNT/seq_index_map.csv  574B
labels_r0.79/work/1LNT/structure_metadata.csv  152B
labels_r0.79/work/1NUV/1nuv.fasta  168B
labels_r0.79/work/1NUV/ions.csv  2KB
labels_r0.79/work/1NUV/per_residue_potential.csv  25KB
labels_r0.79/work/1NUV/residue_renumber_map.csv  2KB
labels_r0.79/work/1NUV/seq_index_map.csv  2KB
labels_r0.79/work/1NUV/structure_metadata.csv  153B
labels_r0.79/work/1Y26/1y26.fasta  80B
labels_r0.79/work/1Y26/ions.csv  663B
labels_r0.79/work/1Y26/per_residue_potential.csv  18KB
labels_r0.79/work/1Y26/residue_renumber_map.csv  1KB
labels_r0.79/work/1Y26/seq_index_map.csv  2KB
labels_r0.79/work/1Y26/structure_metadata.csv  152B
labels_r0.79/work/2OIU/2oiu.fasta  160B
labels_r0.79/work/2OIU/ions.csv  838B
labels_r0.79/work/2OIU/per_residue_potential.csv  36KB
```


### mg_full/mg_sensitivity_summary.txt

```
structures analysed: 84; skipped: 26; failed: 10
chains: 167; median ions per structure: 5; chains kept after dropping solver-mismatched rows

=== mg_r0.79 vs stored (ion-free) labels, within-chain centred ===
  [c1p] chain-median Pearson 0.998 (IQR 0.990..0.999); RMSE of the within-chain change 2.18 kT/e; variance of the label changed 0.5% (median); global offset (mean change) +0.90 kT/e
  [m4] chain-median Pearson 0.988 (IQR 0.953..0.998); RMSE of the within-chain change 1.59 kT/e; variance of the label changed 2.9% (median); global offset (mean change) +0.90 kT/e
  [m8] chain-median Pearson 0.977 (IQR 0.914..0.993); RMSE of the within-chain change 1.72 kT/e; variance of the label changed 6.1% (median); global offset (mean change) +1.00 kT/e
  [m12] chain-median Pearson 0.986 (IQR 0.948..0.994); RMSE of the within-chain change 1.12 kT/e; variance of the label changed 5.0% (median); global offset (mean change) +1.05 kT/e

=== mg0_r0.79 vs stored (ion-free) labels, within-chain centred ===
  [c1p] chain-median Pearson 0.999 (IQR 0.999..1.000); RMSE of the within-chain change 1.16 kT/e; variance of the label changed 0.1% (median); global offset (mean change) -0.24 kT/e
  [m4] chain-median Pearson 0.999 (IQR 0.997..0.999); RMSE of the within-chain change 0.51 kT/e; variance of the label changed 0.3% (median); global offset (mean change) -0.24 kT/e
  [m8] chain-median Pearson 0.999 (IQR 0.998..1.000); RMSE of the within-chain change 0.33 kT/e; variance of the label changed 0.3% (median); global offset (mean change) -0.25 kT/e
  [m12] chain-median Pearson 1.000 (IQR 0.999..1.000); RMSE of the within-chain change 0.19 kT/e; variance of the label changed 0.2% (median); global offset (mean change) -0.19 kT/e

=== mg_r0.79: how the change depends on distance to the nearest Mg2+ (within-chain centred) ===
  [c1p] (0.0, 4.0] A: n=36, rms change 104.03 kT/e (313% of label sd); (4.0, 6.0] A: n=258, rms change 15.78 kT/e (47% of label sd); (6.0, 8.0] A: n=1088, rms change 5.48 kT/e (16% of label sd); (8.0, 12.0] A: n=2369, rms change 3.09 kT/e (9% of label sd); (12.0, 20.0] A: n=2289, rms change 2.68 kT/e (8% of label sd); (20.0, 1000000000.0] A: n=2344, rms change 1.90 kT/e (6% of label sd)
  [m4] (0.0, 4.0] A: n=36, rms change 66.48 kT/e (496% of label sd); (4.0, 6.0] A: n=258, rms change 16.55 kT/e (124% of label sd); (6.0, 8.0] A: n=1088, rms change 5.37 kT/e (40% of label sd); (8.0, 12.0] A: n=2369, rms change 2.47 kT/e (18% of label sd); (12.0, 20.0] A: n=2289, rms change 2.23 kT/e (17% of label sd); (20.0, 1000000000.0] A: n=2346, rms change 1.54 kT/e (11% of label sd)
  [m8] (0.0, 4.0] A: n=36, rms change 23.46 kT/e (230% of label sd); (4.0, 6.0] A: n=258, rms change 11.19 kT/e (110% of label sd); (6.0, 8.0] A: n=1088, rms change 6.32 kT/e (62% of label sd); (8.0, 12.0] A: n=2369, rms change 2.29 kT/e (22% of label sd); (12.0, 20.0] A: n=2289, rms change 1.98 kT/e (19% of label sd); (20.0, 1000000000.0] A: n=2346, rms change 1.51 kT/e (15% of label sd)
  [m12] (0.0, 4.0] A: n=36, rms change 8.59 kT/e (112% of label sd); (4.0, 6.0] A: n=258, rms change 5.06 kT/e (66% of label sd); (6.0, 8.0] A: n=1088, rms change 3.77 kT/e (49% of label sd); (8.0, 12.0] A: n=2369, rms change 1.99 kT/e (26% of label sd); (12.0, 20.0] A: n=2289, rms change 1.48 kT/e (19% of label sd); (20.0, 1000000000.0] A: n=2346, rms change 1.24 kT/e (16% of label sd)

=== control mg0_r0.79 (same cavity, charge 0): this is the NON-charge part of the change ===
  [m8] (0.0, 4.0] A: rms change 0.65 kT/e; (4.0, 6.0] A: rms change 1.32 kT/e; (6.0, 8.0] A: rms change 1.31 kT/e; (8.0, 12.0] A: rms change 0.63 kT/e; (12.0, 20.0] A: rms change 0.37 kT/e; (20.0, 1000000000.0] A: rms change 0.29 kT/e

PLAIN READING (m8, first radius):
  Median within-chain correlation 0.977: keeping Mg2+ barely reorders the residues. Ion-free labels are a harmless simplification for the within-chain audit; state it in one sentence.
```


### mg_full/mg_effect_by_distance.csv  shape=(48, 9)

```
  variant qty                  bin    n  mean_change  sd_change  rms_change  label_sd_pooled  rms_change_over_label_sd
mg0_r0.79 c1p           (0.0, 4.0]   36       -1.059      2.893       3.043           33.269                     0.091
mg0_r0.79 c1p           (4.0, 6.0]  258       -1.156      2.840       3.061           33.269                     0.092
mg0_r0.79 c1p           (6.0, 8.0] 1088       -0.788      2.019       2.167           33.269                     0.065
mg0_r0.79 c1p          (8.0, 12.0] 2369        0.095      1.713       1.715           33.269                     0.052
mg0_r0.79 c1p         (12.0, 20.0] 2289        0.226      1.421       1.438           33.269                     0.043
mg0_r0.79 c1p (20.0, 1000000000.0] 2344        0.192      1.264       1.278           33.269                     0.038
mg0_r0.79 m12           (0.0, 4.0]   36       -0.086      0.272       0.281            7.691                     0.037
mg0_r0.79 m12           (4.0, 6.0]  258       -0.305      0.584       0.658            7.691                     0.086
mg0_r0.79 m12           (6.0, 8.0] 1088       -0.346      0.587       0.682            7.691                     0.089
mg0_r0.79 m12          (8.0, 12.0] 2369       -0.088      0.418       0.427            7.691                     0.056
mg0_r0.79 m12         (12.0, 20.0] 2289        0.145      0.245       0.284            7.691                     0.037
mg0_r0.79 m12 (20.0, 1000000000.0] 2346        0.142      0.179       0.229            7.691                     0.030
mg0_r0.79  m4           (0.0, 4.0]   36       -0.518      1.751       1.802           13.404                     0.134
mg0_r0.79  m4           (4.0, 6.0]  258       -1.272      2.306       2.629           13.404                     0.196
mg0_r0.79  m4           (6.0, 8.0] 1088       -0.817      1.671       1.859           13.404                     0.139
mg0_r0.79  m4          (8.0, 12.0] 2369        0.077      0.660       0.664           13.404                     0.050
mg0_r0.79  m4         (12.0, 20.0] 2289        0.280      0.343       0.442           13.404                     0.033
mg0_r0.79  m4 (20.0, 1000000000.0] 2346        0.176      0.366       0.406           13.404                     0.030
mg0_r0.79  m8           (0.0, 4.0]   36       -0.241      0.617       0.654           10.187                     0.064
mg0_r0.79  m8           (4.0, 6.0]  258       -0.650      1.155       1.323           10.187                     0.130
mg0_r0.79  m8           (6.0, 8.0] 1088       -0.690      1.119       1.314           10.187                     0.129
mg0_r0.79  m8          (8.0, 12.0] 2369       -0.018      0.632       0.632           10.187                     0.062
mg0_r0.79  m8         (12.0, 20.0] 2289        0.247      0.273       0.369           10.187                     0.036
mg0_r0.79  m8 (20.0, 1000000000.0] 2346        0.171      0.230       0.287           10.187                     0.028
 mg_r0.79 c1p           (0.0, 4.0]   36       66.998     80.706     104.026           33.269                     3.127
 mg_r0.79 c1p           (4.0, 6.0]  258        9.659     12.498      15.777           33.269                     0.474
 mg_r0.79 c1p           (6.0, 8.0] 1088        1.821      5.169       5.478           33.269                     0.165
 mg_r0.79 c1p          (8.0, 12.0] 2369       -0.852      2.967       3.086           33.269                     0.093
 mg_r0.79 c1p         (12.0, 20.0] 2289       -1.289      2.353       2.683           33.269                     0.081
 mg_r0.79 c1p (20.0, 1000000000.0] 2344       -0.817      1.716       1.900           33.269                     0.057
 mg_r0.79 m12           (0.0, 4.0]   36        5.798      6.425       8.588            7.691                     1.117
 mg_r0.79 m12           (4.0, 6.0]  258        3.136      3.985       5.064            7.691                     0.658
 mg_r0.79 m12           (6.0, 8.0] 1088        1.864      3.274       3.766            7.691                     0.490
 mg_r0.79 m12          (8.0, 12.0] 2369        0.365      1.952       1.986            7.691                     0.258
 mg_r0.79 m12         (12.0, 20.0] 2289       -0.863      1.204       1.481            7.691                     0.193
 mg_r0.79 m12 (20.0, 1000000000.0] 2346       -0.824      0.924       1.238            7.691                     0.161
 mg_r0.79  m4           (0.0, 4.0]   36       55.128     37.681      66.479           13.404                     4.959
 mg_r0.79  m4           (4.0, 6.0]  258       10.436     12.876      16.555           13.404                     1.235
 mg_r0.79  m4           (6.0, 8.0] 1088        1.992      4.989       5.369           13.404                     0.401
 mg_r0.79  m4          (8.0, 12.0] 2369       -0.800      2.335       2.468           13.404                     0.184
... (8 more rows)
```


## 8. mg_pilot

```
labels_r0.79/all_residue_potentials.csv  324KB
labels_r0.79/work/2OIU/2oiu.fasta  160B
labels_r0.79/work/2OIU/ions.csv  838B
labels_r0.79/work/2OIU/per_residue_potential.csv  36KB
labels_r0.79/work/2OIU/residue_renumber_map.csv  3KB
labels_r0.79/work/2OIU/seq_index_map.csv  3KB
labels_r0.79/work/2OIU/structure_metadata.csv  152B
labels_r0.79/work/2QUW/2quw.fasta  170B
labels_r0.79/work/2QUW/ions.csv  2KB
labels_r0.79/work/2QUW/per_residue_potential.csv  34KB
labels_r0.79/work/2QUW/residue_renumber_map.csv  2KB
labels_r0.79/work/2QUW/seq_index_map.csv  3KB
labels_r0.79/work/2QUW/structure_metadata.csv  152B
labels_r0.79/work/2Z75/2z75.fasta  158B
labels_r0.79/work/2Z75/ions.csv  1008B
labels_r0.79/work/2Z75/per_residue_potential.csv  36KB
labels_r0.79/work/2Z75/residue_renumber_map.csv  3KB
labels_r0.79/work/2Z75/seq_index_map.csv  3KB
labels_r0.79/work/2Z75/structure_metadata.csv  152B
labels_r0.79/work/3Q3Z/3q3z.fasta  166B
labels_r0.79/work/3Q3Z/ions.csv  1KB
labels_r0.79/work/3Q3Z/per_residue_potential.csv  38KB
labels_r0.79/work/3Q3Z/residue_renumber_map.csv  3KB
labels_r0.79/work/3Q3Z/seq_index_map.csv  4KB
labels_r0.79/work/3Q3Z/structure_metadata.csv  153B
labels_r0.79/work/5D5L/5d5l.fasta  340B
labels_r0.79/work/5D5L/ions.csv  4KB
labels_r0.79/work/5D5L/per_residue_potential.csv  77KB
labels_r0.79/work/5D5L/residue_renumber_map.csv  5KB
labels_r0.79/work/5D5L/seq_index_map.csv  7KB
labels_r0.79/work/5D5L/structure_metadata.csv  152B
labels_r0.79/work/5DH6/5dh6.fasta  85B
labels_r0.79/work/5DH6/ions.csv  678B
labels_r0.79/work/5DH6/per_residue_potential.csv  17KB
labels_r0.79/work/5DH6/residue_renumber_map.csv  1KB
labels_r0.79/work/5DH6/seq_index_map.csv  2KB
labels_r0.79/work/5DH6/structure_metadata.csv  153B
labels_r0.79/work/7ELQ/7elq.fasta  108B
labels_r0.79/work/7ELQ/ions.csv  2KB
labels_r0.79/work/7ELQ/per_residue_potential.csv  23KB
```


### mg_pilot/mg_sensitivity_summary.txt

```
structures analysed: 10; skipped: 0; failed: 0
chains: 23; median ions per structure: 7; chains kept after dropping solver-mismatched rows

=== mg_r0.79 vs stored (ion-free) labels, within-chain centred ===
  [c1p] chain-median Pearson 0.995 (IQR 0.986..0.999); RMSE of the within-chain change 3.13 kT/e; variance of the label changed 1.1% (median); global offset (mean change) +1.07 kT/e
  [m4] chain-median Pearson 0.988 (IQR 0.872..0.998); RMSE of the within-chain change 1.97 kT/e; variance of the label changed 2.9% (median); global offset (mean change) +0.77 kT/e
  [m8] chain-median Pearson 0.972 (IQR 0.789..0.993); RMSE of the within-chain change 2.15 kT/e; variance of the label changed 7.2% (median); global offset (mean change) +0.98 kT/e
  [m12] chain-median Pearson 0.990 (IQR 0.907..0.995); RMSE of the within-chain change 1.15 kT/e; variance of the label changed 5.7% (median); global offset (mean change) +0.89 kT/e

=== mg0_r0.79 vs stored (ion-free) labels, within-chain centred ===
  [c1p] chain-median Pearson 1.000 (IQR 0.998..1.000); RMSE of the within-chain change 1.03 kT/e; variance of the label changed 0.1% (median); global offset (mean change) -0.20 kT/e
  [m4] chain-median Pearson 0.999 (IQR 0.998..0.999); RMSE of the within-chain change 0.55 kT/e; variance of the label changed 0.3% (median); global offset (mean change) -0.19 kT/e
  [m8] chain-median Pearson 0.999 (IQR 0.998..1.000); RMSE of the within-chain change 0.26 kT/e; variance of the label changed 0.2% (median); global offset (mean change) -0.17 kT/e
  [m12] chain-median Pearson 1.000 (IQR 0.999..1.000); RMSE of the within-chain change 0.17 kT/e; variance of the label changed 0.2% (median); global offset (mean change) -0.15 kT/e

=== mg_r0.79: how the change depends on distance to the nearest Mg2+ (within-chain centred) ===
  [c1p] (0.0, 4.0] A: n=8, rms change 99.64 kT/e (306% of label sd); (4.0, 6.0] A: n=56, rms change 19.92 kT/e (61% of label sd); (6.0, 8.0] A: n=141, rms change 5.93 kT/e (18% of label sd); (8.0, 12.0] A: n=322, rms change 4.41 kT/e (14% of label sd); (12.0, 20.0] A: n=442, rms change 3.77 kT/e (12% of label sd); (20.0, 1000000000.0] A: n=315, rms change 2.70 kT/e (8% of label sd)
  [m4] (0.0, 4.0] A: n=8, rms change 89.54 kT/e (756% of label sd); (4.0, 6.0] A: n=56, rms change 21.58 kT/e (182% of label sd); (6.0, 8.0] A: n=141, rms change 5.74 kT/e (48% of label sd); (8.0, 12.0] A: n=322, rms change 3.94 kT/e (33% of label sd); (12.0, 20.0] A: n=442, rms change 3.21 kT/e (27% of label sd); (20.0, 1000000000.0] A: n=315, rms change 1.81 kT/e (15% of label sd)
  [m8] (0.0, 4.0] A: n=8, rms change 26.97 kT/e (345% of label sd); (4.0, 6.0] A: n=56, rms change 13.74 kT/e (176% of label sd); (6.0, 8.0] A: n=141, rms change 6.33 kT/e (81% of label sd); (8.0, 12.0] A: n=322, rms change 2.57 kT/e (33% of label sd); (12.0, 20.0] A: n=442, rms change 2.34 kT/e (30% of label sd); (20.0, 1000000000.0] A: n=315, rms change 1.50 kT/e (19% of label sd)
  [m12] (0.0, 4.0] A: n=8, rms change 7.91 kT/e (138% of label sd); (4.0, 6.0] A: n=56, rms change 6.12 kT/e (107% of label sd); (6.0, 8.0] A: n=141, rms change 3.63 kT/e (63% of label sd); (8.0, 12.0] A: n=322, rms change 2.10 kT/e (37% of label sd); (12.0, 20.0] A: n=442, rms change 1.60 kT/e (28% of label sd); (20.0, 1000000000.0] A: n=315, rms change 1.20 kT/e (21% of label sd)

=== control mg0_r0.79 (same cavity, charge 0): this is the NON-charge part of the change ===
  [m8] (0.0, 4.0] A: rms change 1.03 kT/e; (4.0, 6.0] A: rms change 1.38 kT/e; (6.0, 8.0] A: rms change 0.84 kT/e; (8.0, 12.0] A: rms change 0.33 kT/e; (12.0, 20.0] A: rms change 0.25 kT/e; (20.0, 1000000000.0] A: rms change 0.21 kT/e

PLAIN READING (m8, first radius):
  Median within-chain correlation 0.972: keeping Mg2+ barely reorders the residues. Ion-free labels are a harmless simplification for the within-chain audit; state it in one sentence.
```


### mg_pilot/mg_effect_by_distance.csv  shape=(48, 9)

```
  variant qty                  bin   n  mean_change  sd_change  rms_change  label_sd_pooled  rms_change_over_label_sd
mg0_r0.79 c1p           (0.0, 4.0]   8       -1.649      2.047       2.527           32.556                     0.078
mg0_r0.79 c1p           (4.0, 6.0]  56       -1.373      3.352       3.595           32.556                     0.110
mg0_r0.79 c1p           (6.0, 8.0] 141       -0.461      1.479       1.544           32.556                     0.047
mg0_r0.79 c1p          (8.0, 12.0] 322        0.071      1.859       1.857           32.556                     0.057
mg0_r0.79 c1p         (12.0, 20.0] 442        0.181      1.952       1.959           32.556                     0.060
mg0_r0.79 c1p (20.0, 1000000000.0] 315        0.166      2.134       2.137           32.556                     0.066
mg0_r0.79 m12           (0.0, 4.0]   8       -0.146      0.341       0.351            5.740                     0.061
mg0_r0.79 m12           (4.0, 6.0]  56       -0.309      0.553       0.629            5.740                     0.110
mg0_r0.79 m12           (6.0, 8.0] 141       -0.287      0.374       0.471            5.740                     0.082
mg0_r0.79 m12          (8.0, 12.0] 322       -0.075      0.246       0.257            5.740                     0.045
mg0_r0.79 m12         (12.0, 20.0] 442        0.115      0.131       0.175            5.740                     0.030
mg0_r0.79 m12 (20.0, 1000000000.0] 315        0.102      0.136       0.169            5.740                     0.030
mg0_r0.79  m4           (0.0, 4.0]   8       -1.408      2.054       2.382           11.846                     0.201
mg0_r0.79  m4           (4.0, 6.0]  56       -1.258      2.647       2.909           11.846                     0.246
mg0_r0.79  m4           (6.0, 8.0] 141       -0.545      0.878       1.031           11.846                     0.087
mg0_r0.79  m4          (8.0, 12.0] 322        0.064      0.419       0.423           11.846                     0.036
mg0_r0.79  m4         (12.0, 20.0] 442        0.211      0.301       0.368           11.846                     0.031
mg0_r0.79  m4 (20.0, 1000000000.0] 315        0.141      0.313       0.343           11.846                     0.029
mg0_r0.79  m8           (0.0, 4.0]   8       -0.538      0.942       1.032            7.812                     0.132
mg0_r0.79  m8           (4.0, 6.0]  56       -0.677      1.210       1.377            7.812                     0.176
mg0_r0.79  m8           (6.0, 8.0] 141       -0.528      0.661       0.844            7.812                     0.108
mg0_r0.79  m8          (8.0, 12.0] 322       -0.020      0.325       0.325            7.812                     0.042
mg0_r0.79  m8         (12.0, 20.0] 442        0.187      0.160       0.246            7.812                     0.032
mg0_r0.79  m8 (20.0, 1000000000.0] 315        0.128      0.170       0.212            7.812                     0.027
 mg_r0.79 c1p           (0.0, 4.0]   8       81.680     61.012      99.643           32.556                     3.061
 mg_r0.79 c1p           (4.0, 6.0]  56       11.415     16.470      19.918           32.556                     0.612
 mg_r0.79 c1p           (6.0, 8.0] 141        1.328      5.796       5.926           32.556                     0.182
 mg_r0.79 c1p          (8.0, 12.0] 322       -1.573      4.129       4.413           32.556                     0.136
 mg_r0.79 c1p         (12.0, 20.0] 442       -1.630      3.400       3.767           32.556                     0.116
 mg_r0.79 c1p (20.0, 1000000000.0] 315       -0.803      2.584       2.702           32.556                     0.083
 mg_r0.79 m12           (0.0, 4.0]   8        5.618      5.949       7.908            5.740                     1.378
 mg_r0.79 m12           (4.0, 6.0]  56        3.622      4.977       6.119            5.740                     1.066
 mg_r0.79 m12           (6.0, 8.0] 141        2.003      3.032       3.625            5.740                     0.632
 mg_r0.79 m12          (8.0, 12.0] 322        0.416      2.063       2.101            5.740                     0.366
 mg_r0.79 m12         (12.0, 20.0] 442       -0.986      1.263       1.601            5.740                     0.279
 mg_r0.79 m12 (20.0, 1000000000.0] 315       -0.725      0.952       1.195            5.740                     0.208
 mg_r0.79  m4           (0.0, 4.0]   8       74.598     52.945      89.541           11.846                     7.559
 mg_r0.79  m4           (4.0, 6.0]  56       12.631     17.655      21.579           11.846                     1.822
 mg_r0.79  m4           (6.0, 8.0] 141        1.442      5.577       5.741           11.846                     0.485
 mg_r0.79  m4          (8.0, 12.0] 322       -1.574      3.612       3.935           11.846                     0.332
... (8 more rows)
```


## 9a. results_bundle.txt (first 120 lines)

```
################ BENCHMARK COULOMB
chains scored: 0 of 189 (failed: 189)################ BENCHMARK APBS
chains scored: 0 of 189 (failed: 189)################ MG PILOT
structures analysed: 10; skipped: 0; failed: 0
chains: 23; median ions per structure: 7; chains kept after dropping solver-mismatched rows

=== mg_r0.79 vs stored (ion-free) labels, within-chain centred ===
  [c1p] chain-median Pearson 0.995 (IQR 0.986..0.999); RMSE of the within-chain change 3.13 kT/e; variance of the label changed 1.1% (median); global offset (mean change) +1.07 kT/e
  [m4] chain-median Pearson 0.988 (IQR 0.872..0.998); RMSE of the within-chain change 1.97 kT/e; variance of the label changed 2.9% (median); global offset (mean change) +0.77 kT/e
  [m8] chain-median Pearson 0.972 (IQR 0.789..0.993); RMSE of the within-chain change 2.15 kT/e; variance of the label changed 7.2% (median); global offset (mean change) +0.98 kT/e
  [m12] chain-median Pearson 0.990 (IQR 0.907..0.995); RMSE of the within-chain change 1.15 kT/e; variance of the label changed 5.7% (median); global offset (mean change) +0.89 kT/e

=== mg0_r0.79 vs stored (ion-free) labels, within-chain centred ===
  [c1p] chain-median Pearson 1.000 (IQR 0.998..1.000); RMSE of the within-chain change 1.03 kT/e; variance of the label changed 0.1% (median); global offset (mean change) -0.20 kT/e
  [m4] chain-median Pearson 0.999 (IQR 0.998..0.999); RMSE of the within-chain change 0.55 kT/e; variance of the label changed 0.3% (median); global offset (mean change) -0.19 kT/e
  [m8] chain-median Pearson 0.999 (IQR 0.998..1.000); RMSE of the within-chain change 0.26 kT/e; variance of the label changed 0.2% (median); global offset (mean change) -0.17 kT/e
  [m12] chain-median Pearson 1.000 (IQR 0.999..1.000); RMSE of the within-chain change 0.17 kT/e; variance of the label changed 0.2% (median); global offset (mean change) -0.15 kT/e

=== mg_r0.79: how the change depends on distance to the nearest Mg2+ (within-chain centred) ===
  [c1p] (0.0, 4.0] A: n=8, rms change 99.64 kT/e (306% of label sd); (4.0, 6.0] A: n=56, rms change 19.92 kT/e (61% of label sd); (6.0, 8.0] A: n=141, rms change 5.93 kT/e (18% of label sd); (8.0, 12.0] A: n=322, rms change 4.41 kT/e (14% of label sd); (12.0, 20.0] A: n=442, rms change 3.77 kT/e (12% of label sd); (20.0, 1000000000.0] A: n=315, rms change 2.70 kT/e (8% of label sd)
  [m4] (0.0, 4.0] A: n=8, rms change 89.54 kT/e (756% of label sd); (4.0, 6.0] A: n=56, rms change 21.58 kT/e (182% of label sd); (6.0, 8.0] A: n=141, rms change 5.74 kT/e (48% of label sd); (8.0, 12.0] A: n=322, rms change 3.94 kT/e (33% of label sd); (12.0, 20.0] A: n=442, rms change 3.21 kT/e (27% of label sd); (20.0, 1000000000.0] A: n=315, rms change 1.81 kT/e (15% of label sd)
  [m8] (0.0, 4.0] A: n=8, rms change 26.97 kT/e (345% of label sd); (4.0, 6.0] A: n=56, rms change 13.74 kT/e (176% of label sd); (6.0, 8.0] A: n=141, rms change 6.33 kT/e (81% of label sd); (8.0, 12.0] A: n=322, rms change 2.57 kT/e (33% of label sd); (12.0, 20.0] A: n=442, rms change 2.34 kT/e (30% of label sd); (20.0, 1000000000.0] A: n=315, rms change 1.50 kT/e (19% of label sd)
  [m12] (0.0, 4.0] A: n=8, rms change 7.91 kT/e (138% of label sd); (4.0, 6.0] A: n=56, rms change 6.12 kT/e (107% of label sd); (6.0, 8.0] A: n=141, rms change 3.63 kT/e (63% of label sd); (8.0, 12.0] A: n=322, rms change 2.10 kT/e (37% of label sd); (12.0, 20.0] A: n=442, rms change 1.60 kT/e (28% of label sd); (20.0, 1000000000.0] A: n=315, rms change 1.20 kT/e (21% of label sd)

=== control mg0_r0.79 (same cavity, charge 0): this is the NON-charge part of the change ===
  [m8] (0.0, 4.0] A: rms change 1.03 kT/e; (4.0, 6.0] A: rms change 1.38 kT/e; (6.0, 8.0] A: rms change 0.84 kT/e; (8.0, 12.0] A: rms change 0.33 kT/e; (12.0, 20.0] A: rms change 0.25 kT/e; (20.0, 1000000000.0] A: rms change 0.21 kT/e

PLAIN READING (m8, first radius):
  Median within-chain correlation 0.972: keeping Mg2+ barely reorders the residues. Ion-free labels are a harmless simplification for the within-chain audit; state it in one sentence.################ CONTROLS ernierna (pretrained file: /data/rna_electrostats/centered_latent_v3/ernierna/audit_variance_partition.csv)

potential_at_c1prime_kT_e  [controls_geom_coulomb, micro]
  unique_lm        pretrained: +0.003   random: -0.002   family: -0.003 [-0.006..+0.000, n=5]
  shared           pretrained: +0.603   random: +0.368   family: +0.637 [+0.574..+0.742, n=5]
  unique_controls  pretrained: +0.193   random: +0.428   family: +0.201 [+0.138..+0.243, n=5]
  unexplained      pretrained: +0.202   random: +0.206   family: +0.164 [+0.121..+0.189, n=5]

potential_mean_kT_e_12A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: +0.015   random: +0.010   family: -0.003 [-0.010..+0.005, n=5]
  shared           pretrained: +0.448   random: +0.069   family: +0.121 [+0.032..+0.213, n=5]
  unique_controls  pretrained: +0.296   random: +0.674   family: +0.679 [+0.562..+0.763, n=5]
  unexplained      pretrained: +0.241   random: +0.247   family: +0.204 [+0.182..+0.227, n=5]

potential_mean_kT_e_4A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: +0.001   random: -0.006   family: -0.004 [-0.010..+0.000, n=5]
  shared           pretrained: +0.402   random: +0.076   family: +0.158 [+0.095..+0.204, n=5]
  unique_controls  pretrained: +0.345   random: +0.671   family: +0.615 [+0.585..+0.666, n=5]
  unexplained      pretrained: +0.252   random: +0.259   family: +0.231 [+0.210..+0.241, n=5]
################ CONTROLS rnafm (pretrained file: /data/rna_electrostats/centered_latent_v3/rnafm/audit_variance_partition.csv)

potential_at_c1prime_kT_e  [controls_geom_coulomb, micro]
  unique_lm        pretrained: -0.003   random: -0.003   family: -0.003 [-0.006..+0.000, n=5]
  shared           pretrained: +0.560   random: +0.508   family: +0.590 [+0.473..+0.739, n=5]
  unique_controls  pretrained: +0.232   random: +0.283   family: +0.250 [+0.141..+0.358, n=5]
  unexplained      pretrained: +0.211   random: +0.211   family: +0.163 [+0.121..+0.192, n=5]

potential_mean_kT_e_12A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: -0.002   random: +0.004   family: -0.001 [-0.008..+0.003, n=5]
  shared           pretrained: +0.409   random: +0.074   family: +0.083 [+0.022..+0.142, n=5]
  unique_controls  pretrained: +0.358   random: +0.692   family: +0.710 [+0.642..+0.767, n=5]
  unexplained      pretrained: +0.235   random: +0.229   family: +0.209 [+0.183..+0.237, n=5]

potential_mean_kT_e_4A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: -0.003   random: +0.009   family: -0.004 [-0.008..+0.002, n=5]
  shared           pretrained: +0.351   random: +0.085   family: +0.126 [+0.091..+0.180, n=5]
  unique_controls  pretrained: +0.404   random: +0.670   family: +0.644 [+0.584..+0.669, n=5]
  unexplained      pretrained: +0.248   random: +0.236   family: +0.234 [+0.211..+0.255, n=5]
################ CONTROLS splicebert (pretrained file: /data/rna_electrostats/centered_latent_v3/splicebert/audit_variance_partition.csv)

potential_at_c1prime_kT_e  [controls_geom_coulomb, micro]
  unique_lm        pretrained: -0.003   random: -0.006   family: -0.002 [-0.004..+0.000, n=5]
  shared           pretrained: +0.417   random: +0.512   family: +0.494 [+0.442..+0.562, n=5]
  unique_controls  pretrained: +0.379   random: +0.284   family: +0.345 [+0.268..+0.379, n=5]
  unexplained      pretrained: +0.208   random: +0.211   family: +0.164 [+0.122..+0.187, n=5]

potential_mean_kT_e_12A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: +0.014   random: +0.005   family: -0.005 [-0.010..+0.003, n=5]
  shared           pretrained: +0.206   random: +0.074   family: +0.041 [-0.008..+0.111, n=5]
  unique_controls  pretrained: +0.538   random: +0.670   family: +0.758 [+0.664..+0.796, n=5]
  unexplained      pretrained: +0.242   random: +0.251   family: +0.205 [+0.179..+0.223, n=5]

potential_mean_kT_e_4A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: +0.004   random: -0.001   family: -0.002 [-0.004..-0.000, n=5]
  shared           pretrained: +0.199   random: +0.093   family: +0.079 [+0.045..+0.133, n=5]
  unique_controls  pretrained: +0.548   random: +0.654   family: +0.694 [+0.656..+0.717, n=5]
  unexplained      pretrained: +0.249   random: +0.253   family: +0.229 [+0.211..+0.243, n=5]
################ CONTROLS rnabert (pretrained file: /data/rna_electrostats/centered_latent_v3/rnabert/audit_variance_partition.csv)

potential_at_c1prime_kT_e  [controls_geom_coulomb, micro]
  unique_lm        pretrained: -0.002   random: -0.001   family: -0.002 [-0.005..+0.001, n=5]
  shared           pretrained: +0.629   random: +0.631   family: +0.660 [+0.611..+0.739, n=5]
  unique_controls  pretrained: +0.209   random: +0.206   family: +0.191 [+0.139..+0.222, n=5]
  unexplained      pretrained: +0.164   random: +0.163   family: +0.151 [+0.125..+0.169, n=5]

potential_mean_kT_e_12A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: +0.001   random: +0.005   family: -0.005 [-0.010..+0.001, n=5]
  shared           pretrained: +0.111   random: +0.077   family: +0.011 [-0.011..+0.038, n=5]
  unique_controls  pretrained: +0.672   random: +0.705   family: +0.791 [+0.754..+0.827, n=5]
  unexplained      pretrained: +0.216   random: +0.212   family: +0.203 [+0.187..+0.220, n=5]

potential_mean_kT_e_4A  [controls_geom_coulomb, micro]
  unique_lm        pretrained: -0.001   random: -0.002   family: -0.005 [-0.008..-0.002, n=5]
  shared           pretrained: +0.138   random: +0.117   family: +0.070 [+0.044..+0.089, n=5]
  unique_controls  pretrained: +0.629   random: +0.650   family: +0.705 [+0.685..+0.732, n=5]
  unexplained      pretrained: +0.234   random: +0.235   family: +0.229 [+0.211..+0.244, n=5]
```


## 9b. Other small CSVs (names and columns only)

```
centered_latent_v3/ernierna/swap_geom/swap_pls_test_r2.csv  cols=['target', 'pls_test_r2']
centered_latent_v3/rnabert/swap_geom/swap_pls_test_r2.csv  cols=['target', 'pls_test_r2']
centered_latent_v3/rnafm/swap_geom/swap_pls_test_r2.csv  cols=['target', 'pls_test_r2']
centered_latent_v3/splicebert/swap_geom/swap_pls_test_r2.csv  cols=['target', 'pls_test_r2']
characterization/ernierna/expA2_decoder_alignment.csv  cols=['target1', 'target2', 'cosine_similarity', 'angle_deg']
characterization/ernierna/expA_cross_target_transfer.csv  cols=['target', 'is_primary', 'frozen_transfer_test_r2', 'frozen_transfer_test_pearson_r', 'affine_recalibrated_test_r2', 'frozen_direction_recovery_frac', 'fresh_probe_test_r2_same_layer']
characterization/ernierna/expB_pca_dimensionality.csv  cols=['n_pcs', 'raw_test_r2', 'centered_test_r2', 'raw_fraction_of_full', 'centered_fraction_of_full', 'cum_explained_variance']
characterization/ernierna/expC_layerwise_curves.csv  cols=['layer', 'raw_val_r2', 'centered_val_r2', 'residual21mer_val_r2']
characterization/rnabert/expA2_decoder_alignment.csv  cols=['target1', 'target2', 'cosine_similarity', 'angle_deg']
characterization/rnabert/expA_cross_target_transfer.csv  cols=['target', 'is_primary', 'frozen_transfer_test_r2', 'frozen_transfer_test_pearson_r', 'affine_recalibrated_test_r2', 'frozen_direction_recovery_frac', 'fresh_probe_test_r2_same_layer']
characterization/rnabert/expB_pca_dimensionality.csv  cols=['n_pcs', 'raw_test_r2', 'centered_test_r2', 'raw_fraction_of_full', 'centered_fraction_of_full', 'cum_explained_variance']
characterization/rnabert/expC_layerwise_curves.csv  cols=['layer', 'raw_val_r2', 'centered_val_r2', 'residual21mer_val_r2']
characterization/rnafm/expA2_decoder_alignment.csv  cols=['target1', 'target2', 'cosine_similarity', 'angle_deg']
characterization/rnafm/expA_cross_target_transfer.csv  cols=['target', 'is_primary', 'frozen_transfer_test_r2', 'frozen_transfer_test_pearson_r', 'affine_recalibrated_test_r2', 'frozen_direction_recovery_frac', 'fresh_probe_test_r2_same_layer']
characterization/rnafm/expB_pca_dimensionality.csv  cols=['n_pcs', 'raw_test_r2', 'centered_test_r2', 'raw_fraction_of_full', 'centered_fraction_of_full', 'cum_explained_variance']
characterization/rnafm/expC_layerwise_curves.csv  cols=['layer', 'raw_val_r2', 'centered_val_r2', 'residual21mer_val_r2']
characterization/splicebert/expA2_decoder_alignment.csv  cols=['target1', 'target2', 'cosine_similarity', 'angle_deg']
characterization/splicebert/expA_cross_target_transfer.csv  cols=['target', 'is_primary', 'frozen_transfer_test_r2', 'frozen_transfer_test_pearson_r', 'affine_recalibrated_test_r2', 'frozen_direction_recovery_frac', 'fresh_probe_test_r2_same_layer']
characterization/splicebert/expB_pca_dimensionality.csv  cols=['n_pcs', 'raw_test_r2', 'centered_test_r2', 'raw_fraction_of_full', 'centered_fraction_of_full', 'cum_explained_variance']
characterization/splicebert/expC_layerwise_curves.csv  cols=['layer', 'raw_val_r2', 'centered_val_r2', 'residual21mer_val_r2']
clustering/chain_cluster_assignments.csv  cols=['cluster_rep', 'chain_id']
controls/families_kmer.csv  cols=['pdb_id', 'chain', 'family', 'source']
controls/families_kmer_t0.5.csv  cols=['pdb_id', 'chain', 'family', 'source']
controls/families_kmer_t0.7.csv  cols=['pdb_id', 'chain', 'family', 'source']
controls/families_kmer_t0.9.csv  cols=['pdb_id', 'chain', 'family', 'source']
controls_matched/analysis/family_paired_differences.csv  cols=['tag', 'target', 'controls', 'weighting', 'component', 'pretrained_familyout', 'pre_lo', 'pre_hi', 'random_familyout', 'rand_lo']
controls_matched/analysis/length_partition.csv  cols=['model', 'run', 'target', 'controls', 'weighting', 'length_bin', 'n_chains', 'n_structures', 'n_residues', 'component']
controls_matched/analysis/paired_differences.csv  cols=['tag', 'target', 'controls', 'weighting', 'component', 'a_value', 'b_mean', 'b_min', 'b_max', 'n_b_runs']
controls_matched/pretrained/ernierna/spike_in_summary.csv  cols=['target', 'controls', 'weighting', 'component', 'value', 'lo', 'hi', 'injected', 'spike_type', 'control_explained_raw_signal']
controls_matched/pretrained/rnabert/spike_in_summary.csv  cols=['target', 'controls', 'weighting', 'component', 'value', 'lo', 'hi', 'injected', 'spike_type', 'control_explained_raw_signal']
controls_matched/pretrained/rnafm/spike_in_summary.csv  cols=['target', 'controls', 'weighting', 'component', 'value', 'lo', 'hi', 'injected', 'spike_type', 'control_explained_raw_signal']
controls_matched/pretrained/splicebert/spike_in_summary.csv  cols=['target', 'controls', 'weighting', 'component', 'value', 'lo', 'hi', 'injected', 'spike_type', 'control_explained_raw_signal']
coulomb_baseline_structure_summary.csv  cols=['pdb_id', 'n_atoms', 'total_charge', 'n_residues']
dssr_controls_join_report.csv  cols=['pdb_id', 'status', 'n_label_rows', 'n_matched', 'n_resname_mismatch']
mg_full/labels_r0.79/work/1J1U/ions.csv  cols=['pdb_id', 'chain', 'resnum', 'resname', 'x', 'y', 'z', 'occupancy', 'bfactor', 'nearest_chain']
mg_full/labels_r0.79/work/1J1U/per_residue_potential.csv  cols=['pdb_id', 'chain', 'resnum', 'resname', 'x', 'y', 'z', 'potential_mean_kT_e_4A', 'potential_std_kT_e_4A', 'n_grid_points_4A']
mg_full/labels_r0.79/work/1J1U/residue_renumber_map.csv  cols=['pdb_id', 'chain', 'synthetic_resnum', 'original_chain', 'original_resnum', 'insertion_code', 'resname']
mg_full/labels_r0.79/work/1J1U/seq_index_map.csv  cols=['pdb_id', 'chain', 'resnum', 'resname', 'base', 'seq_index', 'original_chain', 'original_resnum', 'insertion_code']
mg_full/labels_r0.79/work/1J1U/structure_metadata.csv  cols=['pdb_id', 'pb_solver', 'charges_possibly_incomplete', 'had_modified_nucleotides', 'resolution_A', 'experimental_method']
mg_full/labels_r0.79/work/1L2X/ions.csv  cols=['pdb_id', 'chain', 'resnum', 'resname', 'x', 'y', 'z', 'occupancy', 'bfactor', 'nearest_chain']
```
