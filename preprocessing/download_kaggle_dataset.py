import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasinpratomo/wound-dataset")

print("Path to dataset files:", path)