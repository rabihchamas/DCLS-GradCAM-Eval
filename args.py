class Args:
    def __init__(self):
        # Valid dcls models: convnext_tiny, convnext_small, convnext_base, caformer_s18, convformer_s18,
        # fastvit_sa24, resnet50
        self.model_name = "convnext_tiny"
        # Valid explainers: "Threshold_GradCAM", "GradCAM"
        self.explainer = "Threshold_GradCAM"
        # If you want to test on all the dataset give 100
        self.using_data_percentage = 100
        self.batch_size = 32
