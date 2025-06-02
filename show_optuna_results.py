import optuna
import os



DB_PATH = "optuna_study.db"
STUDY_NAME = "siamese_unet_tuning_v3" 


def display_study_results(study_name, storage_name):
    """
    Loads an Optuna study and prints the results of completed trials.

    Args:
        study_name (str): The name of the study to load.
        storage_name (str): The storage URL (e.g., 'sqlite:///optuna_study.db').
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"Error: Study '{study_name}' not found in '{storage_name}'.")
        print("Please ensure the STUDY_NAME in the script matches the one used during tuning.")
        print("You can list available studies using the Optuna CLI: optuna studies --storage sqlite:///your_db_file.db")
        return
    except Exception as e:
        print(f"An error occurred while loading the study: {e}")
        return

    print(f"Study name: {study.study_name}")
    print(f"Best trial (based on Optuna's internal tracking for the objective):")
    try:
        best_trial_overall = study.best_trial
        print(f"  Number: {best_trial_overall.number}")
        print(f"  Value (Objective): {best_trial_overall.value}")
        print(f"  Params: ")
        for key, value in best_trial_overall.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("  No completed trials found to determine the best trial.")

    print("\n--- All Completed Trials ---")
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        print("No trials have been completed yet.")
        return

    completed_trials.sort(key=lambda t: t.number)

    for trial in completed_trials:
        print(f"\nTrial Number: {trial.number}")
        print(f"  State: {trial.state}")
        print(f"  Value (Objective): {trial.value}") # This is the value returned by your objective function
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    storage_url = f"sqlite:///{DB_PATH}"
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_PATH}' not found.")
        print("Please ensure the path is correct and the database file exists in the current directory or specify the full path.")
    else:
        print(f"Attempting to load study '{STUDY_NAME}' from '{storage_url}'...")
        display_study_results(study_name=STUDY_NAME, storage_name=storage_url)