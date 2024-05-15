import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

class SequentialGenerator:
    def __init__(self, data):
        self.data = data.copy()  # Utiliser une copie pour éviter de modifier l'original
        self.one_hot_groups = self.identify_one_hot_groups()  # Identifier les groupes one-hot encodés

    def identify_one_hot_groups(self):
        one_hot_groups = {}
        columns = self.data.columns
        for col in columns:
            if '_' in col:
                base_col = col.rsplit('_', 1)[0]
                if base_col not in one_hot_groups:
                    one_hot_groups[base_col] = [col]
                else:
                    one_hot_groups[base_col].append(col)
        return one_hot_groups

    def add_noise_to_variable(self, value, noise_scale=1):
        noise = np.random.normal(loc=0, scale=noise_scale)
        if isinstance(value, (bool, np.bool_)):
            return not value if np.abs(noise) > noise_scale / 2 else value
        elif isinstance(value, (int, np.integer)):
            if noise > 0.5 * noise_scale:
                return int(np.ceil(value + noise))
            else:
                return int(np.floor(value + noise))
        else:
            return value + noise

    def sample_from_class_interval(self, conditioned_data, num_classes=10):
        if isinstance(conditioned_data.iloc[0], (bool, np.bool_)):
            # Pour les booléens, calculer les probabilités en fonction de leur fréquence conditionnelle
            value_counts = conditioned_data.value_counts(normalize=True)
            return np.random.choice(value_counts.index, p=value_counts.values)
        elif isinstance(conditioned_data.iloc[0], (int, np.integer)):
            # Pour les entiers, utiliser l'histogramme mais arrondir les résultats
            if len(conditioned_data) < num_classes:
                return int(np.random.choice(conditioned_data))
            
            counts, bin_edges = np.histogram(conditioned_data, bins=num_classes)
            bin_probs = counts / counts.sum()
            chosen_bin = np.random.choice(range(len(bin_probs)), p=bin_probs)
            sampled_value = np.random.uniform(bin_edges[chosen_bin], bin_edges[chosen_bin + 1])
            return int(round(sampled_value))
        else:
            # Pour les autres types, utiliser l'histogramme normalement
            if len(conditioned_data) < num_classes:
                return np.random.choice(conditioned_data)
            
            counts, bin_edges = np.histogram(conditioned_data, bins=num_classes)
            bin_probs = counts / counts.sum()
            chosen_bin = np.random.choice(range(len(bin_probs)), p=bin_probs)
            sampled_value = np.random.uniform(bin_edges[chosen_bin], bin_edges[chosen_bin + 1])
            return sampled_value

    def generate_data(self, target_variable_name, noise_scale=1, num_classes=10, n_samples=1):
        new_data = pd.DataFrame(columns=self.data.columns)
        
        for _ in range(n_samples):
            generated_sample = {}
            generated_variables = set()  # Conserver les variables générées

            target_value = np.random.choice(self.data[target_variable_name])
            noisy_target_value = self.add_noise_to_variable(target_value, noise_scale)
            generated_sample[target_variable_name] = noisy_target_value
            generated_variables.add(target_variable_name)
            input_variable_name = target_variable_name

            while len(generated_sample) < len(self.data.columns):
                next_variable_name = self.find_next_variable(input_variable_name, generated_variables)
                if next_variable_name is None:
                    break
                generated_variables.add(next_variable_name)

                if next_variable_name in self.one_hot_groups:
                    # Pour les variables one-hot encodées, échantillonner parmi les catégories
                    base_col = next_variable_name.rsplit('_', 1)[0]
                    group_cols = self.one_hot_groups[base_col]
                    group_data = self.data[group_cols]
                    
                    if isinstance(self.data[input_variable_name].iloc[0], (bool, np.bool_)):
                        conditioned_data = group_data[self.data[input_variable_name] == noisy_target_value]
                    else:
                        bin_edges = np.histogram_bin_edges(self.data[input_variable_name], bins=num_classes)
                        class_idx = np.digitize([noisy_target_value], bin_edges)[0] - 1
                        class_idx = max(0, min(class_idx, len(bin_edges) - 2))
                        class_min = bin_edges[class_idx]
                        class_max = bin_edges[class_idx + 1]
                        conditioned_data = group_data[(self.data[input_variable_name] >= class_min) & (self.data[input_variable_name] < class_max)]

                    if conditioned_data.empty:
                        break

                    sampled_row = conditioned_data.sample(n=1, replace=True).iloc[0]
                    for col in group_cols:
                        generated_sample[col] = sampled_row[col]

                else:
                    if isinstance(self.data[input_variable_name].iloc[0], (bool, np.bool_)):
                        conditioned_data = self.data[self.data[input_variable_name] == noisy_target_value][next_variable_name]
                    else:
                        bin_edges = np.histogram_bin_edges(self.data[input_variable_name], bins=num_classes)
                        class_idx = np.digitize([noisy_target_value], bin_edges)[0] - 1
                        class_idx = max(0, min(class_idx, len(bin_edges) - 2))
                        class_min = bin_edges[class_idx]
                        class_max = bin_edges[class_idx + 1]
                        conditioned_data = self.data[(self.data[input_variable_name] >= class_min) & (self.data[input_variable_name] < class_max)][next_variable_name]

                    if conditioned_data.empty:
                        break

                    sampled_value = self.sample_from_class_interval(conditioned_data, num_classes)
                    generated_sample[next_variable_name] = sampled_value

                input_variable_name = next_variable_name
                noisy_target_value = generated_sample[next_variable_name]

            new_data = pd.concat([new_data, pd.DataFrame([generated_sample])], ignore_index=True)

        return new_data

    def find_next_variable(self, target_variable_name, exclude_variables):
        input_variable_names = [col for col in self.data.columns if col != target_variable_name and col not in exclude_variables]
        if not input_variable_names:
            return None
        target_variable = self.data[target_variable_name]
        mutual_info = {col: mutual_info_regression(self.data[[col]], target_variable)[0] for col in input_variable_names}
        next_variable_name = max(mutual_info, key=mutual_info.get)
        return next_variable_name