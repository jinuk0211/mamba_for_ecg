!pip install timesfm
import timesfm

tfm = timesfm.TimesFm(
    context_len=512,
    horizon_len=1,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='gpu',
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
