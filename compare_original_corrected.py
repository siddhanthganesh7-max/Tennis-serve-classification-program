"""Compares the laboratory's originally reported ROM values with the corrected values (Figure 1, Table S2).
Usage: python compare_original_corrected.py serves70_corrected.csv <lab report csv> mot_source_check.csv"""
import sys, numpy as np, pandas as pd
c=pd.read_csv(sys.argv[1]); lab=pd.read_csv(sys.argv[2]); src=pd.read_csv(sys.argv[3])
ks=["EL_flex","EL_prosup","LB_ext","LB_bend","LB_rot","R_HI_flex","R_HI_abd","R_HI_rot","R_KN_flex","R_AN_flex","R_AN_inv","L_HI_flex","L_HI_abd","L_HI_rot","L_KN_flex","L_AN_flex","L_AN_inv"]
m=c.merge(lab[["serve_id","bad_capture"]+ks],on="serve_id",suffixes=("","_lab")).merge(src[["serve_id","own_recording"]],on="serve_id").rename(columns={"own_recording":"joint_data_is_own"})
m["mad"]=np.median(np.abs(m[ks].values-m[[k+"_lab" for k in ks]].values),axis=1)
print(m.groupby("joint_data_is_own").mad.describe().round(1)); print(m.groupby("trc").mad.median().round(1))
m.to_csv("compare_lab_vs_corrected.csv",index=False)
