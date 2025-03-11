import joblib

model = joblib.load("F:\DEM_GLAMM\DEM_CREATION_FINAL\correction_model\GLAM_LKO_DemCorrection_RF_model_gridsearch_v3_20230329.joblib")
oob_score = model.oob_score_

print(oob_score)

quit()