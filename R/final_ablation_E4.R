# Leave-one-group-out ablation with E4 (640 deployable features) as the reference.
# Every variant uses the same settings as the reference. Groups are disjoint; each column of E4 is
# assigned to the first group it matches. Thresholds come from 5-fold out-of-fold training scores.
source("R/lib_triage.R")

D <- load_triage()
fs <- feature_sets(names(D$X))
g4 <- fs$g4
tr <- D$idx; te <- setdiff(seq_len(nrow(D$X)), tr)
yt <- D$y[te]

grp <- rep(NA_character_, length(g4)); names(grp) <- g4
assign_grp <- function(cols, name) { cols <- intersect(cols, g4); grp[cols[is.na(grp[cols])]] <<- name }
assign_grp(vitals_cols, "vitals")
assign_grp(grep("^(insurance_|residence_|region_|race_|sex_)", g4, value = TRUE), "demographics")
assign_grp(grep("^(ems_arrival_|episode_|vitals_during_visit_|seen_last_72h_|no_payment_|is_injury_poison_)", g4, value = TRUE), "administrative")
assign_grp(fs$comorb, "comorbidities")
assign_grp(fs$text, "complaint_text")
grp[is.na(grp)] <- "engineered_clinical"
print(table(grp))
fwrite(data.table(column = g4, group = unname(grp)), file.path(OUTDIR, "ablation_groups.csv"))

run_case <- function(label, cols) {
  t0 <- Sys.time()
  raw <- predict_ordinal(fit_ordinal(D$X, D$y, tr, cols, 600), D$X, te, cols)
  th <- find_thresholds(raw, yt)$thresholds
  pc <- decode(raw, th)
  append_csv(metrics_row(label, length(cols), yt, pc, list(t1 = th[1], t2 = th[2], t3 = th[3], t4 = th[4])),
             "ablation_E4.csv")
  if (label == "E4_full") save_pred("E4_ablation_ref", te, yt, pc, raw)
  cat(sprintf("%s done in %.0fs\n", label, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  print(fread(file.path(OUTDIR, "ablation_E4.csv")))
}

run_case("E4_full", g4)
for (g in c("complaint_text", "administrative", "demographics", "engineered_clinical", "comorbidities", "vitals"))
  run_case(paste0("minus_", g), g4[grp != g])
cat("DONE\n")
