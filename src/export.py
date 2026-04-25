import pandas as pd
from pathlib import Path

# OUTPUT directory
OUTPUT_DIR = Path("OUTPUT")

# #

# Save predictions to CSV
def save_predictions(y_true, y_pred, filename="predictions.csv"):

    """ 
    Saves model predictions to a structured CSV file inside OUTPUT
    """

    # Ensure OUTPUT directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build results dataframe
    results_dataframe = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    # Write to disk (relative path)
    output_path = OUTPUT_DIR / filename
    results_dataframe.to_csv(output_path, index=False)

    # #
# #    