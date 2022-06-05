# !/bin/sh
saved_model_cli show --all --dir model_weights/tfkeras_model >> info/tfkeras_model.txt
saved_model_cli show --all --dir model_weights/TRTFP32 >> info/TRTFP32.txt
saved_model_cli show --all --dir model_weights/TRTFP16 >> info/TRTFP16.txt
saved_model_cli show --all --dir model_weights/TRTINT8 >> info/TRTINT.txt
