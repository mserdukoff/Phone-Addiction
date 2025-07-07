import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Set the tracking URI to match the training script
mlflow.set_tracking_uri("file:./mlruns")

# Retrieve experiment by name
experiment_name = "TeenPhoneAddiction"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    raise RuntimeError(f"❌ Experiment '{experiment_name}' not found. Did you run the training script?")

EXPERIMENT_ID = experiment.experiment_id

# Get latest run in the experiment
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=[EXPERIMENT_ID],
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError("❌ No runs found in experiment. Did you run train_model.py?")

latest_run = runs[0]
model_uri = f"runs:/{latest_run.info.run_id}/model"

# Load model
model = mlflow.sklearn.load_model(model_uri)

def predict(features):
    return model.predict([features])[0]
