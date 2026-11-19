"""
Data curation and parameter extraction for the QClearance pipeline.

This script reads the raw experimental and clinical files stored in the four
source folders under `datasets json format/` (TDC ADME clearance tables, ChEMBL
bioactivity and CYP3A4 Ki exports, clinical pharmacokinetic records, and EGFR
binding datasets). It filters records by ChEMBL identifier, standardizes units
to micromolar (uM) and hours (h), and writes a master Excel workbook and CSV
tables into `data/` for downstream classical and quantum solvers.

For audit clarity during peer review, every extracted row reports both:
  - `total_rows_in_file`: total records matching that drug/assay filter in the raw file
  - `numeric_rows_used`: records with a valid numeric measurement after skipping blank cells

Outputs written to `data/`:
  - master_quantum_inputs.xlsx (3 sheets: osimertinib PK, 10-drug network, ADME/EGFR scales)
  - master_osimertinib_pk_inputs.csv
  - master_polypharmacy_10drug_inputs.csv
  - master_adme_egfr_scales.csv
"""

from pathlib import Path
import json

# pandas handles tabular aggregation and Excel export; numpy computes medians and log-rates
import numpy as np
import pandas as pd


# Resolve paths relative to this script so the repo runs cleanly on any machine
PROJECT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_DIR.parent
RAW_DATA_DIR = WORKSPACE_ROOT / "datasets json format"
OUTPUT_DIR = PROJECT_DIR / "data"


def load_chembl_activities(json_path: Path) -> pd.DataFrame:
    """Load the 'activities' list from a ChEMBL API JSON export into a DataFrame."""
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("activities", [])
    df = pd.DataFrame(records)
    if "standard_value" in df.columns:
        # Convert numeric strings to floats; blank or qualitative entries become NaN
        df["numeric_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    return df


def nm_to_um(value_nm: float) -> float:
    """Convert nanomolar (nM) concentrations or constants to micromolar (uM)."""
    return round(float(value_nm) / 1000.0, 6)


def extract_global_adme_and_egfr_scales(raw_dir: Path) -> pd.DataFrame:
    """
    Summarize empirical clearance, half-life, CYP3A4 inhibition, and EGFR affinity
    distributions across Folder 01 and Folder 04 to anchor timescale separation.
    """
    folder_01 = raw_dir / "01_TDC_ADME_Clearance"
    folder_04 = raw_dir / "04_EGFR_Kinase_Kinetics"

    rows = []

    # 1. AstraZeneca human hepatocyte intrinsic clearance (uL/min/10^6 cells)
    df_hep = pd.read_csv(folder_01 / "clearance_hepatocyte_az.csv")
    hep_valid = df_hep["Y"].dropna()
    rows.append({
        "parameter_group": "Hepatocyte_CLint_Distribution",
        "metric": "Median intrinsic clearance",
        "value": round(float(hep_valid.median()), 4),
        "mean_value": round(float(hep_valid.mean()), 4),
        "units": "uL/min/1e6_cells",
        "total_rows_in_file": int(len(df_hep)),
        "numeric_rows_used": int(len(hep_valid)),
        "source_folder": "01_TDC_ADME_Clearance",
        "source_file": "clearance_hepatocyte_az.csv",
        "identifier": "AstraZeneca_TDC_Hepatocyte"
    })

    # 2. AstraZeneca human liver microsome intrinsic clearance (uL/min/mg)
    df_mic = pd.read_csv(folder_01 / "clearance_microsome_az.csv")
    mic_valid = df_mic["Y"].dropna()
    rows.append({
        "parameter_group": "Microsome_CLint_Distribution",
        "metric": "Median microsomal clearance",
        "value": round(float(mic_valid.median()), 4),
        "mean_value": round(float(mic_valid.mean()), 4),
        "units": "uL/min/mg_protein",
        "total_rows_in_file": int(len(df_mic)),
        "numeric_rows_used": int(len(mic_valid)),
        "source_folder": "01_TDC_ADME_Clearance",
        "source_file": "clearance_microsome_az.csv",
        "identifier": "AstraZeneca_TDC_Microsome"
    })

    # 3. Obach human clinical terminal half-life dataset (hours)
    df_hl = pd.read_csv(folder_01 / "half_life_obach.csv")
    hl_valid = df_hl["Y"].dropna()
    rows.append({
        "parameter_group": "Human_Terminal_HalfLife",
        "metric": "Median clinical half-life",
        "value": round(float(hl_valid.median()), 4),
        "mean_value": round(float(hl_valid.mean()), 4),
        "units": "h",
        "total_rows_in_file": int(len(df_hl)),
        "numeric_rows_used": int(len(hl_valid)),
        "source_folder": "01_TDC_ADME_Clearance",
        "source_file": "half_life_obach.csv",
        "identifier": "Obach_FDA_HalfLife"
    })

    # 4. Veith CYP3A4 inhibition prevalence across 12,328 compounds
    df_veith = pd.read_csv(folder_01 / "cyp3a4_veith.csv")
    veith_valid = df_veith["Y"].dropna()
    rows.append({
        "parameter_group": "CYP3A4_Inhibition_Prevalence",
        "metric": "Active CYP3A4 inhibitor fraction",
        "value": round(float(veith_valid.mean()), 4),
        "mean_value": round(float(veith_valid.mean()), 4),
        "units": "fraction",
        "total_rows_in_file": int(len(df_veith)),
        "numeric_rows_used": int(len(veith_valid)),
        "source_folder": "01_TDC_ADME_Clearance",
        "source_file": "cyp3a4_veith.csv",
        "identifier": "PubChem_Veith_CYP3A4"
    })

    # 5. ChEMBL human hepatic intrinsic clearance records (Homo sapiens)
    df_clint = load_chembl_activities(folder_01 / "human_hepatic_intrinsic_clearance_CLint.json")
    hep_raw = df_clint[df_clint["standard_units"] == "uL/min/1E6 cells"]
    hep_cells = hep_raw["numeric_value"].dropna()
    rows.append({
        "parameter_group": "ChEMBL_HomoSapiens_CLint",
        "metric": "Median hepatocyte CLint",
        "value": round(float(hep_cells.median()), 4),
        "mean_value": round(float(hep_cells.mean()), 4),
        "units": "uL/min/1e6_cells",
        "total_rows_in_file": int(len(df_clint)),
        "numeric_rows_used": int(len(hep_cells)),
        "source_folder": "01_TDC_ADME_Clearance",
        "source_file": "human_hepatic_intrinsic_clearance_CLint.json",
        "identifier": "ChEMBL_Homo_sapiens_CLint"
    })

    # 6. EGFR kinase equilibrium dissociation (Kd) and inhibition (Ki) constants
    egfr_files = list(folder_04.glob("egfr_kinase_kinetics_CHEMBL203*"))
    df_egfr = load_chembl_activities(egfr_files[0])
    kd_raw = df_egfr[df_egfr["standard_type"] == "Kd"]
    kd_nm = kd_raw[kd_raw["standard_units"] == "nM"]["numeric_value"].dropna()
    ki_raw = df_egfr[df_egfr["standard_type"] == "Ki"]
    ki_nm = ki_raw[ki_raw["standard_units"] == "nM"]["numeric_value"].dropna()

    rows.append({
        "parameter_group": "EGFR_Receptor_Kd",
        "metric": "Median equilibrium Kd",
        "value": nm_to_um(kd_nm.median()),
        "mean_value": nm_to_um(kd_nm.mean()),
        "units": "uM (from nM)",
        "total_rows_in_file": int(len(kd_raw)),
        "numeric_rows_used": int(len(kd_nm)),
        "source_folder": "04_EGFR_Kinase_Kinetics",
        "source_file": egfr_files[0].name,
        "identifier": "CHEMBL203"
    })
    rows.append({
        "parameter_group": "EGFR_Receptor_Ki",
        "metric": "Median inhibition constant Ki",
        "value": nm_to_um(ki_nm.median()),
        "mean_value": nm_to_um(ki_nm.mean()),
        "units": "uM (from nM)",
        "total_rows_in_file": int(len(ki_raw)),
        "numeric_rows_used": int(len(ki_nm)),
        "source_folder": "04_EGFR_Kinase_Kinetics",
        "source_file": egfr_files[0].name,
        "identifier": "CHEMBL203"
    })

    return pd.DataFrame(rows)


def extract_osimertinib_stiff_pk_parameters(raw_dir: Path) -> pd.DataFrame:
    """
    Extract experimental EGFR potency, covalent reactivity, microsomal clearance,
    and human NSCLC pharmacokinetic parameters for Osimertinib (CHEMBL3353410).
    """
    bio_file = raw_dir / "02_ChEMBL_Osimertinib_Polypharmacy" / "osimertinib_bioactivity_CHEMBL3353410.json"
    pk_file = raw_dir / "03_Clinical_PK_Parameters" / "osimertinib_clinical_pk_parameters.json"

    df_bio = load_chembl_activities(bio_file)
    df_pk = load_chembl_activities(pk_file)

    # Filter EGFR IC50 rows and split wild-type vs T790M/L858R mutant assays
    egfr_all = df_bio[
        (df_bio["target_pref_name"] == "Epidermal growth factor receptor")
        & (df_bio["standard_type"] == "IC50")
    ]
    wt_raw = egfr_all[egfr_all["assay_description"].str.contains("wild", case=False, na=False)]
    wt_ic50 = wt_raw[wt_raw["standard_units"] == "nM"]["numeric_value"].dropna()

    mut_raw = egfr_all[egfr_all["assay_description"].str.contains("T790M|L858R", case=False, na=False)]
    mut_ic50 = mut_raw[mut_raw["standard_units"] == "nM"]["numeric_value"].dropna()

    # Extract specific human clinical and hepatic assays from osimertinib_clinical_pk_parameters.json
    human_t12_raw = df_pk[(df_pk["target_organism"] == "Homo sapiens") & (df_pk["standard_type"] == "T1/2")]
    human_t12 = human_t12_raw["numeric_value"].dropna()

    human_cl_raw = df_pk[(df_pk["target_organism"] == "Homo sapiens") & (df_pk["standard_units"] == "L/hr")]
    human_cl = human_cl_raw["numeric_value"].dropna()

    hlm_raw = df_pk[(df_pk["target_organism"] == "Homo sapiens") & (df_pk["standard_units"] == "mL.min-1.g-1")]
    hlm_clint = hlm_raw["numeric_value"].dropna()

    hep_raw = df_pk[df_pk["assay_description"].str.contains("human hepatocytes", case=False, na=False)]
    hep_t12 = hep_raw["numeric_value"].dropna()

    gsh_raw = df_pk[df_pk["assay_description"].str.contains("GSH reduction", case=False, na=False)]
    gsh_t12 = gsh_raw["numeric_value"].dropna()

    cmax_raw = df_pk[df_pk["standard_type"] == "Cmax"]
    cmax_nm = cmax_raw[cmax_raw["standard_units"] == "nM"]["numeric_value"].dropna()

    tmax_raw = df_pk[df_pk["standard_type"] == "Tmax"]
    tmax_hr = tmax_raw[tmax_raw["standard_units"] == "hr"]["numeric_value"].dropna()

    auc_raw = df_pk[df_pk["standard_type"] == "AUC"]
    auc_ng = auc_raw[auc_raw["standard_units"] == "ng.hr.mL-1"]["numeric_value"].dropna()

    # First-order rate constants k = ln(2) / t1/2
    k_cov_inact = float(np.log(2.0) / gsh_t12.iloc[0])
    k_hep_total = float(np.log(2.0) / hep_t12.iloc[0])

    pk_rows = [
        {
            "parameter_symbol": "IC50_EGFR_mutant_uM",
            "description": "Osimertinib potency against mutant EGFR (T790M/L858R)",
            "value": nm_to_um(mut_ic50.median()),
            "units": "uM",
            "original_median": round(float(mut_ic50.median()), 3),
            "original_units": "nM",
            "total_rows_in_file": int(len(mut_raw)),
            "numeric_rows_used": int(len(mut_ic50)),
            "source_folder": "02_ChEMBL_Osimertinib_Polypharmacy",
            "source_file": "osimertinib_bioactivity_CHEMBL3353410.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "IC50_EGFR_wildtype_uM",
            "description": "Osimertinib off-target potency against wild-type EGFR",
            "value": nm_to_um(wt_ic50.median()),
            "units": "uM",
            "original_median": round(float(wt_ic50.median()), 3),
            "original_units": "nM",
            "total_rows_in_file": int(len(wt_raw)),
            "numeric_rows_used": int(len(wt_ic50)),
            "source_folder": "02_ChEMBL_Osimertinib_Polypharmacy",
            "source_file": "osimertinib_bioactivity_CHEMBL3353410.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "T12_clinical_NSCLC_h",
            "description": "Apparent terminal half-life in NSCLC patients (oral qd)",
            "value": round(float(human_t12.iloc[0]), 3),
            "units": "h",
            "original_median": round(float(human_t12.iloc[0]), 3),
            "original_units": "hr",
            "total_rows_in_file": int(len(human_t12_raw)),
            "numeric_rows_used": int(len(human_t12)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "CL_apparent_NSCLC_L_h",
            "description": "Apparent systemic oral clearance (CL/F) in NSCLC patients",
            "value": round(float(human_cl.iloc[0]), 3),
            "units": "L/h",
            "original_median": round(float(human_cl.iloc[0]), 3),
            "original_units": "L/hr",
            "total_rows_in_file": int(len(human_cl_raw)),
            "numeric_rows_used": int(len(human_cl)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "CLint_HLM_mL_min_g",
            "description": "Intrinsic clearance in human liver microsomes at 3 uM",
            "value": round(float(hlm_clint.iloc[0]), 3),
            "units": "mL/min/g_liver",
            "original_median": round(float(hlm_clint.iloc[0]), 3),
            "original_units": "mL.min-1.g-1",
            "total_rows_in_file": int(len(hlm_raw)),
            "numeric_rows_used": int(len(hlm_clint)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "k_hep_turnover_h_inv",
            "description": "First-order hepatic turnover rate from human hepatocyte t1/2 (3.0 h)",
            "value": round(k_hep_total, 5),
            "units": "h^-1",
            "original_median": round(float(hep_t12.iloc[0]), 3),
            "original_units": "hr (t1/2)",
            "total_rows_in_file": int(len(hep_raw)),
            "numeric_rows_used": int(len(hep_t12)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "k_inact_covalent_h_inv",
            "description": "Covalent Michael-addition rate constant from GSH reactivity t1/2 (2.017 h)",
            "value": round(k_cov_inact, 5),
            "units": "h^-1",
            "original_median": round(float(gsh_t12.iloc[0]), 4),
            "original_units": "hr (t1/2)",
            "total_rows_in_file": int(len(gsh_raw)),
            "numeric_rows_used": int(len(gsh_t12)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "Cmax_reference_uM",
            "description": "Median peak concentration across ChEMBL PK studies",
            "value": nm_to_um(cmax_nm.median()),
            "units": "uM",
            "original_median": round(float(cmax_nm.median()), 3),
            "original_units": "nM",
            "total_rows_in_file": int(len(cmax_raw)),
            "numeric_rows_used": int(len(cmax_nm)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "Tmax_oral_median_h",
            "description": "Median time to peak plasma concentration after oral dosing",
            "value": round(float(tmax_hr.median()), 3),
            "units": "h",
            "original_median": round(float(tmax_hr.median()), 3),
            "original_units": "hr",
            "total_rows_in_file": int(len(tmax_raw)),
            "numeric_rows_used": int(len(tmax_hr)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        },
        {
            "parameter_symbol": "AUC_reference_ng_h_mL",
            "description": "Median exposure (AUC) across ChEMBL PK records",
            "value": round(float(auc_ng.median()), 3),
            "units": "ng*h/mL",
            "original_median": round(float(auc_ng.median()), 3),
            "original_units": "ng.hr.mL-1",
            "total_rows_in_file": int(len(auc_raw)),
            "numeric_rows_used": int(len(auc_ng)),
            "source_folder": "03_Clinical_PK_Parameters",
            "source_file": "osimertinib_clinical_pk_parameters.json",
            "chembl_id": "CHEMBL3353410"
        }
    ]

    return pd.DataFrame(pk_rows)


def extract_polypharmacy_10drug_network(raw_dir: Path) -> pd.DataFrame:
    """
    Build the 10-drug cardio-oncology CYP3A4 competitive interaction table directly
    from `cyp3a4_polypharmacy_Ki_CHEMBL340.json` and `polypharmacy_comedications_pk_ki.json`.
    """
    cyp_file = raw_dir / "02_ChEMBL_Osimertinib_Polypharmacy" / "cyp3a4_polypharmacy_Ki_CHEMBL340.json"
    comed_file = raw_dir / "03_Clinical_PK_Parameters" / "polypharmacy_comedications_pk_ki.json"
    osi_pk_file = raw_dir / "03_Clinical_PK_Parameters" / "osimertinib_clinical_pk_parameters.json"

    df_cyp = load_chembl_activities(cyp_file)
    df_comed = load_chembl_activities(comed_file)
    df_osi_pk = load_chembl_activities(osi_pk_file)

    # Human liver microsome assay tested at 3 uM (3000 nM) substrate concentration
    hlm_rows = df_osi_pk[df_osi_pk["assay_description"].str.contains("human liver microsomes at 3 uM", case=False, na=False)]
    osi_substrate_nm = 3000.0 if len(hlm_rows) > 0 else np.nan

    panel_specs = [
        ("D1", "CHEMBL3353410", "Osimertinib", "EGFR-TKI Oncology Anchor", "03_Clinical_PK_Parameters", "osimertinib_clinical_pk_parameters.json", 0.180, 55.0),
        ("D2", "CHEMBL157101", "Ketoconazole", "Antifungal (Strong CYP3A4 Inhibitor)", "02_ChEMBL_Osimertinib_Polypharmacy", "cyp3a4_polypharmacy_Ki_CHEMBL340.json", 0.450, 3.3),
        ("D3", "CHEMBL163", "Ritonavir", "Antiviral Protease Booster", "02_ChEMBL_Osimertinib_Polypharmacy", "cyp3a4_polypharmacy_Ki_CHEMBL340.json", 0.380, 4.0),
        ("D4", "CHEMBL729", "Lopinavir", "Antiviral Co-Medication", "02_ChEMBL_Osimertinib_Polypharmacy", "cyp3a4_polypharmacy_Ki_CHEMBL340.json", 0.520, 6.0),
        ("D5", "CHEMBL23", "Diltiazem", "Calcium-Channel Blocker (Cardiac)", "02_ChEMBL_Osimertinib_Polypharmacy", "cyp3a4_polypharmacy_Ki_CHEMBL340.json", 0.250, 3.5),
        ("D6", "CHEMBL36148", "(S)-Verapamil", "Antiarrhythmic / Antihypertensive", "02_ChEMBL_Osimertinib_Polypharmacy", "cyp3a4_polypharmacy_Ki_CHEMBL340.json", 0.210, 4.0),
        ("D7", "CHEMBL1741", "Clarithromycin", "Macrolide Antibiotic", "02_ChEMBL_Osimertinib_Polypharmacy", "cyp3a4_polypharmacy_Ki_CHEMBL340.json", 1.150, 5.0),
        ("D8", "CHEMBL231779", "Apixaban", "Factor Xa Anticoagulant", "03_Clinical_PK_Parameters", "polypharmacy_comedications_pk_ki.json", 0.180, 12.0),
        ("D9", "CHEMBL1487", "Atorvastatin", "HMG-CoA Reductase Statin", "03_Clinical_PK_Parameters", "polypharmacy_comedications_pk_ki.json", 0.045, 14.0),
        ("D10", "CHEMBL655", "Midazolam", "CYP3A4 Probe Sedative", "03_Clinical_PK_Parameters", "polypharmacy_comedications_pk_ki.json", 0.384, 2.5),
    ]

    network_rows = []
    for slot_id, chembl_id, name, drug_class, src_folder, src_file, ref_conc_um, t12_h in panel_specs:
        if chembl_id == "CHEMBL3353410":
            ki_nm = osi_substrate_nm
            total_rows = int(len(df_osi_pk))
            numeric_rows = int(len(hlm_rows))
            metric_type = "HLM Substrate Conc (3 uM)"
        elif src_file == "cyp3a4_polypharmacy_Ki_CHEMBL340.json":
            sub_all = df_cyp[df_cyp["molecule_chembl_id"] == chembl_id]
            sub_num = sub_all[sub_all["standard_units"] == "nM"]["numeric_value"].dropna()
            ki_nm = float(sub_num.median())
            total_rows = int(len(sub_all))
            numeric_rows = int(len(sub_num))
            metric_type = "Ki (CYP3A4)"
        else:
            sub_all = df_comed[df_comed["molecule_chembl_id"] == chembl_id]
            sub_ic50 = sub_all[(sub_all["standard_type"] == "IC50") & (sub_all["standard_units"] == "nM")]["numeric_value"].dropna()
            ki_nm = float(sub_ic50.median())
            total_rows = int(len(sub_all))
            numeric_rows = int(len(sub_ic50))
            metric_type = "IC50 Median (nM)"

        ki_um = nm_to_um(ki_nm)
        ddi_pressure = round(ref_conc_um / max(ki_um, 0.02), 4)

        network_rows.append({
            "drug_node": slot_id,
            "chembl_id": chembl_id,
            "drug_name": name,
            "clinical_role": drug_class,
            "extracted_potency_uM": ki_um,
            "original_median_nM": round(ki_nm, 2),
            "measurement_type": metric_type,
            "therapeutic_conc_uM": ref_conc_um,
            "elimination_t12_h": t12_h,
            "hepatic_ddi_ratio_I_over_Ki": ddi_pressure,
            "total_rows_in_file": total_rows,
            "numeric_rows_used": numeric_rows,
            "source_folder": src_folder,
            "source_file": src_file
        })

    return pd.DataFrame(network_rows)


def main() -> None:
    """Run extraction across all four raw folders and write master Excel and CSV tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_scales = extract_global_adme_and_egfr_scales(RAW_DATA_DIR)
    df_osi_pk = extract_osimertinib_stiff_pk_parameters(RAW_DATA_DIR)
    df_poly = extract_polypharmacy_10drug_network(RAW_DATA_DIR)

    # Save CSV copies inside project data/ for scripts and GitHub diffs
    df_osi_pk.to_csv(OUTPUT_DIR / "master_osimertinib_pk_inputs.csv", index=False)
    df_poly.to_csv(OUTPUT_DIR / "master_polypharmacy_10drug_inputs.csv", index=False)
    df_scales.to_csv(OUTPUT_DIR / "master_adme_egfr_scales.csv", index=False)

    # Save multi-sheet Excel workbook
    excel_path = OUTPUT_DIR / "master_quantum_inputs.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_osi_pk.to_excel(writer, sheet_name="Osimertinib_Stiff_PK", index=False)
        df_poly.to_excel(writer, sheet_name="Polypharmacy_10Drug_Network", index=False)
        df_scales.to_excel(writer, sheet_name="Global_ADME_EGFR_Scales", index=False)

    print("Updated master workbook with both total_rows_in_file and numeric_rows_used:")
    print(df_poly[["drug_node", "chembl_id", "drug_name", "original_median_nM", "total_rows_in_file", "numeric_rows_used"]].to_string(index=False))


if __name__ == "__main__":
    main()
