

from


# Valid dcls models: convnext_tiny, convnext_small, convnext_base, caformer_s18, convformer_s18, fastvit_sa24

model_name = "model_name"
Xmethod = thresholding_GradCAM


#Print interpretability score
evaluate(model_name,Xmethod, DCLS=True)