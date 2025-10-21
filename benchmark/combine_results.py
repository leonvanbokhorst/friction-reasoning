import json
import glob
import os


def combine_eval_results():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    output_file = os.path.join(results_dir, "all_evals.json")

    all_data = []

    # Find all eval-*.json files
    eval_files = sorted(glob.glob(os.path.join(results_dir, "eval-*.json")))

    if not eval_files:
        print("No eval files found to combine.")
        return

    print(f"Found {len(eval_files)} files to combine.")

    # Read and aggregate data from each file
    for file_path in eval_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {file_path}")

    # Write the combined data to the output file
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Successfully combined all eval files into {output_file}")


if __name__ == "__main__":
    combine_eval_results()
