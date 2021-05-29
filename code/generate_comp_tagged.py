from LLM import LLM
if __name__ == '__main__':
    data_path=""
    L = LLM(0, 0, 4999, data_path,save_files_prefix="mod1", optim_lambda_val=0.25)
    L.train()
    L.predict("comp1.words","comp_m1_312088727_212172027.wtag")
    L = LLM(0, 0, 249, data_path,save_files_prefix="mod2", optim_lambda_val=0.25)
    L.train()
    L.predict("comp2.words","comp_m2_312088727_212172027.wtag")
