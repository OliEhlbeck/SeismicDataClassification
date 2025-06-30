import json
import utils

datas = {
    "hy": [
        "DATA/hy_jan",
        "DATA/hy_may"
    ],
    "rf": [
        "DATA/rf_jan",
        "DATA/rf_may"
    ],
}

class_labels = {
    "hy": 0,
    "rf": 1
}

def compute_features(data: list):
    from scipy.stats import kurtosis
    from scipy.stats import skew
    import numpy as np

    kurt = kurtosis(data)
    skewness = skew(data)
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    rms = np.sqrt(np.mean(np.square(data)))
    variance = np.var(data)
    range_val = max_val - min_val
    return [kurt, skewness, mean, median, std, min_val, max_val, rms, variance, range_val]

if __name__ == "__main__":
    use_z_only = False
    print("Processing dataset sources...")
    dataset = []
    for classname, dataset_months in datas.items():

        for ds_months in dataset_months:
            ds_events = utils.list_content_of_directory(ds_months)['directories']
            print(f"\tProcessing {classname} {ds_months}...")
            for event in ds_events:
                full_path = f"{ds_months}/{event}"
                data_files = utils.list_content_of_directory(full_path)['files']

                if use_z_only == False:
                    #Var 1: Use all three components

                    data_e = utils.read_ascii_file(f"{full_path}/MBGA_E.ASC")
                    data_n = utils.read_ascii_file(f"{full_path}/MBGA_N.ASC")
                    data_z = utils.read_ascii_file(f"{full_path}/MBGA_Z.ASC")

                    data_merged = [[e, n, z] for e, n, z in zip(data_e, data_n, data_z)]

                    #  Remove invalid data samples. Invalid samples are marked with [0, 0, 0]
                    data_merged = [e for e in data_merged if e != [0, 0, 0]]

                    if len(data_merged) == 0:
                        continue

                    # Compute other features using sklearn
                    features_e = compute_features([e[0] for e in data_merged])
                    features_n = compute_features([n[1] for n in data_merged])
                    features_z = compute_features([z[2] for z in data_merged])



                    # Add features to dataset_merged
                    for sample in data_merged:
                        sample.extend(features_e)
                        sample.extend(features_n)
                        sample.extend(features_z)
                        sample.append(class_labels[classname])

                else:
                    ##Var 2: Use Z-component only.

                    data_z = utils.read_ascii_file(f"{full_path}/MBGA_Z.ASC")

                    data_merged = [[z] for z in  data_z]

                    #  Remove invalid data samples. Invalid samples are marked with [0, 0, 0]
                    data_merged = [e for e in data_merged if e != [0]]

                    if len(data_merged) == 0:
                        continue

                    # Compute other features using sklearn
                    features_z = compute_features([z[0] for z in data_merged])


                    # Add features to dataset_merged
                    for sample in data_merged:
                        sample.extend(features_z)
                        sample.append(class_labels[classname])


                # Add session-data to dataset_class
                dataset.extend(data_merged)

    # Scale each element in list that is float to [0, 1] in dataset_merged
    data_scaled = []
    for idx in range(len(dataset[0])):
        data = [e[idx] for e in dataset]
        min_e, max_e = min(data), max(data)

        # Scale data
        data = [(e - min_e) / (max_e - min_e) for e in data]

        data_scaled.append(data)

    # Update dataset_merged. Zip data_scaled as list of lists
    dataset = list(zip(*data_scaled))

    # Write dataset_class to file
    target_folder = "dataset"
    if use_z_only:
        output_file = f"{target_folder}/dataset_z.json"
    else:
        output_file = f"{target_folder}/dataset_all.json"
    print(f"Saving dataset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)




