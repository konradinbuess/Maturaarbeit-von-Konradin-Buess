import hashlib
import threading
from pathlib import Path
from typing import cast
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy

plt.rcParams['figure.figsize'] = (8.0, 4.0)
plt.rcParams['figure.dpi'] = 150


def format_results(weights):
    if len(weights) < 8:
        weights = np.append(weights, 1 / np.prod(weights).item())
    formatted_results = (
        f"Aussenpolitik: {weights[0]:.3f}, Wirtschaft: {weights[1]:.3f}, "
        f"Finanzen: {weights[2]:.3f}, Law&Order: {weights[3]:.3f}, "
        f"Migration: {weights[4]:.3f}, Umweltschutz: {weights[5]:.3f}, "
        f"Sozialstaat: {weights[6]:.3f}, Gesellschaft: {weights[7]:.3f}"
    )
    return formatted_results


def politicians_groups_and_matrix(politicians=None, start_question=41, end_question=3116):
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    cache_folder = Path("cache")
    cache_folder.mkdir(parents=True, exist_ok=True)
    politicians_path = cache_folder / f"politicians_{politicians_hash}_{start_question}_{end_question}.pkl"
    if politicians_path.exists():
        pol, pol_yes_groups, pol_no_groups, indices = joblib.load(politicians_path)
    else:
        print("Politiker und Stimmgruppen werden berechnet")
        excel_tab = pd.read_excel(
            r'C:\Users\konra\OneDrive - DIG-IT - eduBS\Dokumente\Gymnasium\MATURAARBEIT\Datei fuer Python.xlsx',
            header=None)
        data_politicians_matrix = excel_tab.iloc[20:28, 12:215].values.astype(float)
        data_politicians_matrix = np.round(data_politicians_matrix, decimals=3)
        pol = data_politicians_matrix[:, politicians]
        pol_yes_groups = {}
        pol_no_groups = {}
        for questions_index in range(start_question, end_question):
            yes_group = set()
            no_group = set()
            for politicians_index in politicians:
                cell_value = cast(str, excel_tab.iloc[questions_index, politicians_index + 12])
                cell_string = str(cell_value).strip().upper()
                if cell_string == "JA":
                    yes_group.add(politicians_index)
                elif cell_string == "NEIN":
                    no_group.add(politicians_index)
            if len(yes_group) > 0 and len(no_group) > 0:
                pol_yes_groups[questions_index - 41] = yes_group
                pol_no_groups[questions_index - 41] = no_group
        joblib.dump((pol, pol_yes_groups, pol_no_groups, politicians), politicians_path)
        print("Politiker und Stimmgruppen sind fertig berechnet")
    return pol, pol_yes_groups, pol_no_groups


def compute_shared_vote_ratio_matrix(politicians=None, start_question=41, end_question=3116):
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    length = len(politicians)
    shared_vote_ratio_matrix = np.ones((length, length))
    politicians_matrix, group_yes_dict, group_no_dict = politicians_groups_and_matrix(
        politicians=politicians, start_question=start_question, end_question=end_question)
    cache_folder = Path("cache")
    cache_folder.mkdir(parents=True, exist_ok=True)
    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    shared_vote_ratio_path = cache_folder / f"shared_vote_ratio_matrix_{politicians_hash}_{start_question}_{end_question}.pkl"
    if shared_vote_ratio_path.exists():
        shared_vote_ratio_matrix = joblib.load(shared_vote_ratio_path)
    else:
        print("shared_vote_ratio_matrix wird berechnet")
        for idx_a in range(length):
            for idx_b in range(idx_a + 1, length):
                count_same_votes = 0
                count_different_votes = 0
                for group_id in group_yes_dict:
                    if group_id in group_no_dict:
                        if idx_a in group_yes_dict[group_id] and idx_b in \
                                group_no_dict[group_id]:
                            count_different_votes += 1
                        elif idx_a in group_no_dict[group_id] and idx_b in \
                                group_yes_dict[group_id]:
                            count_different_votes += 1
                    if idx_a in group_yes_dict[group_id] and idx_b in \
                            group_yes_dict[group_id]:
                        count_same_votes += 1
                for group_id in group_no_dict:
                    if idx_a in group_no_dict[group_id] and idx_b in \
                            group_no_dict[group_id]:
                        count_same_votes += 1
                if count_same_votes + count_different_votes != 0:
                    shared_vote_ratio = count_same_votes / (count_same_votes + count_different_votes)
                else:
                    shared_vote_ratio = np.nan
                shared_vote_ratio_matrix[idx_a, idx_b] = shared_vote_ratio
                shared_vote_ratio_matrix[idx_b, idx_a] = shared_vote_ratio
        joblib.dump(shared_vote_ratio_matrix, shared_vote_ratio_path)
        print("shared_vote_ratio_matrix wurde fertig berechnet")
    return shared_vote_ratio_matrix


def extract_boundary_vectors(politicians=None, p=1.0,
                             weighted_by_votes=True, start_question=41,
                             end_question=3116, permute_axes=None, zero_axes=None):
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    if permute_axes is None:
        permute_axes = [False] * 8
    permuted_hash = hashlib.md5(str(permute_axes).encode('utf-8')).hexdigest()
    if zero_axes is None:
        zero_axes = [False] * 8
    else:
        zero_axes = [bool(x) for x in zero_axes]
    zero_hash = f"_{hashlib.md5(str(zero_axes).encode('utf-8')).hexdigest()}" if any(zero_axes) else ""
    if weighted_by_votes:
        weighted = "weighted"
    else:
        weighted = "not_weighted"
    cache_folder = Path("cache")
    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    boundary_path = cache_folder / f"boundary_vectors_{politicians_hash}_{start_question}_{end_question}_{p}_{weighted}_{permuted_hash}{zero_hash}.pkl"
    if boundary_path.exists():
        boundary_vectors, vector_counts = joblib.load(boundary_path)
    else:
        print("boundary_vectors werden berechnet")
        full_shared_vote_ratio_matrix = compute_shared_vote_ratio_matrix(politicians=None,
                                                                         start_question=start_question,
                                                                         end_question=end_question)
        full_politicians_matrix = \
        politicians_groups_and_matrix(politicians=None, start_question=start_question,
                                      end_question=end_question)[0].copy()
        for i, permute_active in enumerate(permute_axes):
            if permute_active:
                np.random.shuffle(full_politicians_matrix[i])
        full_difference_tensor = full_politicians_matrix.T[:, np.newaxis, :] - full_politicians_matrix.T[
            np.newaxis, :, :]
        full_difference_tensor = abs(full_difference_tensor) ** p
        distance_tensor = full_difference_tensor[politicians, :, :]
        shared_vote_ratio_matrix = full_shared_vote_ratio_matrix[politicians, :]
        distance_a_b_tensor = distance_tensor[:, :, np.newaxis, :]
        distance_a_c_tensor = distance_tensor[:, np.newaxis, :, :]
        distance_difference_tensor = distance_a_b_tensor - distance_a_c_tensor
        shared_vote_ratio_a_b_tensor = shared_vote_ratio_matrix[:, :, np.newaxis]
        shared_vote_ratio_a_c_tensor = shared_vote_ratio_matrix[:, np.newaxis, :]
        shared_vote_ratio_difference_tensor = shared_vote_ratio_a_c_tensor - shared_vote_ratio_a_b_tensor
        boundary_vectors_tensor = distance_difference_tensor * np.sign(shared_vote_ratio_difference_tensor)[
            :, :, :, np.newaxis]
        # Für die Fälle das A, B, C nicht alle eine gemeinsame abstimmung hatten, steht nun im Tensor ein NaN.
        # Dies wird im folgenden mit 0 ersetzt, wodurch der Vektor für die Zielfunktion verschwindet.
        boundary_vectors_tensor = np.nan_to_num(boundary_vectors_tensor, nan=0.0)
        valid_indices_mask = np.triu(np.ones((203, 203), dtype=bool), k=1)[np.newaxis, :, :]
        a_indices = np.array(politicians)[:, np.newaxis, np.newaxis]
        b_indices = np.arange(203)[np.newaxis, :, np.newaxis]
        c_indices = np.arange(203)[np.newaxis, np.newaxis, :]
        valid_indices_mask = valid_indices_mask & (a_indices != b_indices) & (a_indices != c_indices)
        boundary_vectors_matrix = boundary_vectors_tensor[valid_indices_mask]
        for i, zero_active in enumerate(zero_axes):
            if zero_active:
                boundary_vectors_matrix[:, i] = boundary_vectors_matrix[:, i] * 0
        prechosen_positiv_mask = np.all(boundary_vectors_matrix >= 0, axis=1) & np.any(boundary_vectors_matrix > 0,
                                                                                       axis=1)
        prechosen_zero_mask = np.all(boundary_vectors_matrix == 0, axis=1)
        prechosen_mask = prechosen_positiv_mask | prechosen_zero_mask
        boundary_vectors_matrix = boundary_vectors_matrix[~prechosen_mask]
        length_vector = np.linalg.norm(boundary_vectors_matrix, axis=1, keepdims=True)
        boundary_vectors_matrix = boundary_vectors_matrix / length_vector

        boundary_vectors, inverse_indices = np.unique(boundary_vectors_matrix, axis=0, return_inverse=True)
        if weighted_by_votes:
            shared_vote_ratio_per_vector = abs(shared_vote_ratio_difference_tensor)[valid_indices_mask]
            weight_per_vector = shared_vote_ratio_per_vector[~prechosen_mask]
        else:
            weight_per_vector = np.ones(boundary_vectors_matrix.shape[0])
        vector_counts = np.bincount(inverse_indices, weight_per_vector)
        joblib.dump((boundary_vectors, vector_counts), boundary_path)
        print("boundary_vectors sind fertig berechnet")
    return boundary_vectors, vector_counts


def compute_error_function(weight=np.ones(8), politicians=None, p=1.0,
                           weighted_by_votes=True,
                           start_question=41, end_question=3116, permute_axes=None, zero_axes=None):
    boundary_vectors, vector_counts = extract_boundary_vectors(politicians=politicians, p=p,
                                                               weighted_by_votes=weighted_by_votes,
                                                               start_question=start_question,
                                                               end_question=end_question,
                                                               permute_axes=permute_axes,
                                                               zero_axes=zero_axes)
    dot_product_vector = np.dot(boundary_vectors, weight)
    mask = dot_product_vector <= 0
    number_of_negativ_dot_product = np.sum(vector_counts[mask])
    return number_of_negativ_dot_product


def minimizer(politicians=None, p=1.0, weighted_by_votes=True,
              start_question=41, end_question=3116, permute_axes=None,
              number_of_repetitions=1000, number_of_points=10001, search_radius=1.0,
              force_optimize_again=False, zero_axes=None):
    if zero_axes is None:
        zero_axes = [False] * 8
    else:
        zero_axes = [bool(x) for x in zero_axes]
    zero_hash = f"_{hashlib.md5(str(zero_axes).encode('utf-8')).hexdigest()}" if any(zero_axes) else ""
    zero_axes_array = np.array(zero_axes, dtype=bool)
    real_number_of_points = number_of_points * search_radius
    if real_number_of_points % 2 == 0:
        real_number_of_points += 1
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    if permute_axes is None:
        permute_axes = [False] * 8
    permuted_hash = hashlib.md5(str(permute_axes).encode('utf-8')).hexdigest()
    if weighted_by_votes:
        weighted = "weighted"
    else:
        weighted = "not_weighted"
    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    cache_name_minimizer = f"{politicians_hash}_{start_question}_{end_question}_{p}_{weighted}_{permuted_hash}{zero_hash}"
    cache_folder = Path("cache")
    minimizer_folder = cache_folder / f"minimizer {cache_name_minimizer}"
    arg_min_path = minimizer_folder / f"argmin.pkl"
    if arg_min_path.exists() and not force_optimize_again:
        arg_min = joblib.load(arg_min_path)
        return arg_min
    else:
        print("Minimum wird berechnet")
        boundary_vectors, vector_counts = extract_boundary_vectors(politicians=politicians, p=p,
                                                                   weighted_by_votes=weighted_by_votes,
                                                                   start_question=start_question,
                                                                   end_question=end_question,
                                                                   permute_axes=permute_axes, zero_axes=zero_axes)
        # Die Einstellungen zum Gebrauch des onnx-Modells wurde mithilfe von Gemini programmiert.
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = True
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        onnx_session = ort.InferenceSession("gpu_matrix_fusion.onnx", sess_options=opts, providers=providers)
        search_radius = np.clip(search_radius, 0.0, 1.0)
        cache_folder = Path("cache")
        minimizer_folder = cache_folder / f"minimizer {cache_name_minimizer}"
        minimizer_folder.mkdir(parents=True, exist_ok=True)
        arg_min_path = minimizer_folder / f"argmin.pkl"
        minimum_path = minimizer_folder / f"minimum.pkl"
        if arg_min_path.exists():
            start_point = joblib.load(arg_min_path)
            print(start_point)
        else:
            start_point = np.ones(8)
            start_point[zero_axes_array] = 0.0
        start_vektor = (start_point / np.sum(start_point)) * 8.0
        joblib.dump(start_vektor, arg_min_path)
        min_value = compute_error_function(weight=start_point,
                                           politicians=politicians, p=p,
                                           weighted_by_votes=weighted_by_votes,
                                           start_question=start_question,
                                           end_question=end_question,
                                           permute_axes=permute_axes,
                                           zero_axes=zero_axes)
        joblib.dump(min_value, minimum_path)
        boundary_vectors_transponiert = np.ascontiguousarray(boundary_vectors.astype(np.float32).T)
        vector_weights = vector_counts.astype(np.float32)
        aktuelles_minimum = joblib.load(minimum_path)
        aktuelles_arg_min = joblib.load(arg_min_path)
        for repetition_index in range(number_of_repetitions):
            lam_array = np.linspace(-search_radius, search_radius,
                                    np.round(real_number_of_points).astype(int))
            zufallsgewicht: np.ndarray = np.random.uniform(-8.0, 8.0, size=8) #type: ignore
            zufallsgewicht[zero_axes_array] = 0.0
            if np.sum(zufallsgewicht) == 0:
                zufallsgewicht = np.ones(8, dtype=np.float32)
                zufallsgewicht[zero_axes_array] = 0.0
            weight_projection = (aktuelles_arg_min / float(np.sum(aktuelles_arg_min))) * 8.0
            projektion_zufallsgewicht = (zufallsgewicht / np.sum(zufallsgewicht)) * 8.0
            zufallsrichtungsvektor: np.ndarray = projektion_zufallsgewicht - weight_projection
            zwischenvektoren_matrix = weight_projection[np.newaxis, :] + lam_array[:, np.newaxis] * \
                                      zufallsrichtungsvektor[np.newaxis, :]
            zwischenvektoren_projektion_matrix = (zwischenvektoren_matrix / np.sum(zwischenvektoren_matrix,
                                                                                   axis=1,
                                                                                   keepdims=True)) * 8.0
            ungueltige_zeilen_maske = np.any(
                (zwischenvektoren_projektion_matrix > 8.0) | (zwischenvektoren_projektion_matrix < -8.0),
                axis=1)
            zwischenvektoren_matrix_bereinigt = zwischenvektoren_projektion_matrix[~ungueltige_zeilen_maske]
            if zwischenvektoren_matrix_bereinigt.shape[0] == 0:
                continue
            matrix = zwischenvektoren_matrix_bereinigt.astype(np.float32)
            chunk_size_points = 1000
            chunk_size_vectors = 100000
            num_sections_points = max(1, matrix.shape[0] // chunk_size_points)
            num_sections_vectors = max(1, boundary_vectors_transponiert.shape[1] // chunk_size_vectors)
            chunks_matrix = np.array_split(matrix, num_sections_points, axis=0)
            chunks_boundary = np.array_split(boundary_vectors_transponiert, num_sections_vectors, axis=1)
            chunks_counts = np.array_split(vector_weights, num_sections_vectors)
            all_partial_results = []
            for chunk_m in chunks_matrix:
                partial_res_for_rows = np.zeros(chunk_m.shape[0], dtype=np.float32)
                for chunk_b, chunk_c in zip(chunks_boundary, chunks_counts):
                    onnx_inputs = {"chunk_matrix": chunk_m, "boundary_matrix": chunk_b,
                                   "vector_counts": chunk_c}
                    onnx_outputs = onnx_session.run(None, onnx_inputs)[0]
                    partial_res_for_rows += onnx_outputs
                all_partial_results.append(partial_res_for_rows)
            model_error_vector = np.concatenate(all_partial_results, axis=0)
            neues_minimum = model_error_vector[np.argmin(model_error_vector)]
            neues_arg_min = zwischenvektoren_matrix_bereinigt[np.argmin(model_error_vector)]
            if neues_minimum < aktuelles_minimum:
                print("Neues Minimum gefunden nach Wiederholung ", repetition_index, " bei ", neues_arg_min,
                      " mit Wert: ", neues_minimum)
                joblib.dump(neues_minimum, minimum_path)
                joblib.dump(neues_arg_min, arg_min_path)
                aktuelles_minimum = neues_minimum
                aktuelles_arg_min = neues_arg_min
            else:
                print("Kein neues Minimum gefunden in Wiederholung: ", repetition_index)
        arg_min = joblib.load(arg_min_path)
        return arg_min


def slice_appending(politicians=None, p=1.0, weighted_by_votes=True,
                    start_question=41, end_question=3116, permute_axes=None,
                    number_of_slices=1000, number_of_points=10001,
                    number_of_repetitions_minimizer=1000, number_of_points_minimizer=10001,
                    search_radius_minimizer=1.0, force_optimize_again=False, zero_axes=None):
    if number_of_points % 2 == 0:
        number_of_points += 1
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    if permute_axes is None:
        permute_axes = [False] * 8
    permuted_hash = hashlib.md5(str(permute_axes).encode('utf-8')).hexdigest()
    if zero_axes is None:
        zero_axes = [False] * 8
    else:
        zero_axes = [bool(x) for x in zero_axes]
    zero_hash = f"{hashlib.md5(str(zero_axes).encode('utf-8')).hexdigest()}_" if any(zero_axes) else ""
    zero_axes_array = np.array(zero_axes, dtype=bool)
    if weighted_by_votes:
        weighted = "weighted"
    else:
        weighted = "not_weighted"
    arg_min = minimizer(politicians=politicians, p=p,
                        weighted_by_votes=weighted_by_votes,
                        start_question=start_question, end_question=end_question,
                        permute_axes=permute_axes,
                        number_of_repetitions=number_of_repetitions_minimizer,
                        number_of_points=number_of_points_minimizer,
                        search_radius=search_radius_minimizer,
                        force_optimize_again=force_optimize_again,
                        zero_axes=zero_axes_array)
    boundary_vectors, vector_counts = extract_boundary_vectors(politicians=politicians, p=p,
                                                               weighted_by_votes=weighted_by_votes,
                                                               start_question=start_question,
                                                               end_question=end_question,
                                                               permute_axes=permute_axes, zero_axes=zero_axes)
    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    cache_name_linear_slices = (
        f"{politicians_hash}_"
        f"{start_question}_"
        f"{end_question}_"
        f"{p}_"
        f"{weighted}_"
        f"{permuted_hash}_"
        f"{zero_hash}"
        f"{number_of_points}"
    )
    # Die Einstellungen zum Gebrauch des onnx-Modells wurde mithilfe von Gemini programmiert.
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = False
    opts.enable_cpu_mem_arena = True
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    onnx_session = ort.InferenceSession("gpu_matrix_fusion.onnx", sess_options=opts, providers=providers)
    cache_folder = Path("cache")
    linear_path_slices_folder = cache_folder / f"linear path slices {cache_name_linear_slices}"
    linear_path_slices_folder.mkdir(parents=True, exist_ok=True)
    model_error_vector_list_path = linear_path_slices_folder / f"model_error_vector_list"
    dot_product_vector_list_path = linear_path_slices_folder / f"dot_product_vetcor_list"
    vector_matrix_list_path = linear_path_slices_folder / f"vector_matrix_list"
    if model_error_vector_list_path.exists():
        model_error_vector_list = joblib.load(model_error_vector_list_path)
    else:
        model_error_vector_list = []
    if dot_product_vector_list_path.exists():
        dot_product_vector_list = joblib.load(dot_product_vector_list_path)
    else:
        dot_product_vector_list = []
    if vector_matrix_list_path.exists():
        vector_matrix_list = joblib.load(vector_matrix_list_path)
    else:
        vector_matrix_list = []
    boundary_vectors_transponiert = np.ascontiguousarray(boundary_vectors.astype(np.float32).T)
    vector_weights = vector_counts.astype(np.float32)

    if np.linalg.norm(arg_min) == 0:
        print("Gewicht darf nicht 0 sein und wurde nun auf 1 gesetzt")
        arg_min = np.ones(8, dtype=np.float32)
        arg_min[zero_axes_array] = 0
    assert arg_min is not None
    weight_projection = np.asarray((arg_min / float(np.sum(arg_min))) * 8.0)
    start_length = len(model_error_vector_list)
    speicher_thread = None
    index_number_of_slices = start_length
    while index_number_of_slices < number_of_slices:
        norm_weight = float(np.linalg.norm(weight_projection))
        lam_array = np.linspace(-1.0, 1.0, number_of_points)
        zufallsgewicht: np.ndarray = np.random.uniform(-8.0, 8.0, size=8) # type: ignore
        zufallsgewicht[zero_axes_array] = 0
        if np.all(zufallsgewicht == 0):
            zufallsgewicht = np.ones(8, dtype=np.float32)
            zufallsgewicht[zero_axes_array] = 0
        projektion_zufallsgewicht = (zufallsgewicht / np.sum(zufallsgewicht)) * 8.0
        zufallsrichtungsvektor = projektion_zufallsgewicht - weight_projection
        zwischenvektoren_matrix = weight_projection[np.newaxis, :] + lam_array[:, np.newaxis] * \
                                  zufallsrichtungsvektor[np.newaxis, :]
        zwischenvektoren_projektion_matrix = (zwischenvektoren_matrix / np.sum(zwischenvektoren_matrix,
                                                                               axis=1,
                                                                               keepdims=True)) * 8.0
        ungueltige_zeilen_maske = np.any(
            (zwischenvektoren_projektion_matrix > 8.0) | (zwischenvektoren_projektion_matrix < -8.0),
            axis=1)
        zwischenvektoren_matrix_bereinigt = zwischenvektoren_projektion_matrix[~ungueltige_zeilen_maske]
        lam_array_bereinigt = lam_array[~ungueltige_zeilen_maske]
        if len(lam_array_bereinigt) < 0.5 * number_of_points:
            continue
        index_number_of_slices += 1
        zwischenvektoren_norm = np.linalg.norm(zwischenvektoren_matrix_bereinigt, axis=1)
        dot_product_vektor = np.dot(zwischenvektoren_matrix_bereinigt, weight_projection)
        dot_product_vektor = dot_product_vektor / (zwischenvektoren_norm * norm_weight)
        dot_product_vektor = np.clip(dot_product_vektor, -1.0, 1.0)
        dot_product_vektor = 1.0 - dot_product_vektor
        negativ_mask = lam_array_bereinigt < 0
        dot_product_vektor = np.where(negativ_mask, -dot_product_vektor, dot_product_vektor)
        dot_product_vector_list.append(dot_product_vektor)
        matrix = zwischenvektoren_matrix_bereinigt.astype(np.float32)
        chunk_size_points = 10000
        chunk_size_vectors = 100000
        num_sections_points = max(1, matrix.shape[0] // chunk_size_points)
        num_sections_vectors = max(1, boundary_vectors_transponiert.shape[1] // chunk_size_vectors)
        chunks_matrix = np.array_split(matrix, num_sections_points, axis=0)
        chunks_boundary = np.array_split(boundary_vectors_transponiert, num_sections_vectors, axis=1)
        chunks_counts = np.array_split(vector_weights, num_sections_vectors)
        all_partial_results = []
        for chunk_m in chunks_matrix:
            partial_res_for_rows = np.zeros(chunk_m.shape[0], dtype=np.float32)
            for chunk_b, chunk_c in zip(chunks_boundary, chunks_counts):
                onnx_inputs = {"chunk_matrix": chunk_m, "boundary_matrix": chunk_b,
                               "vector_counts": chunk_c}
                onnx_outputs = onnx_session.run(None, onnx_inputs)[0]
                partial_res_for_rows += onnx_outputs
            all_partial_results.append(partial_res_for_rows)
        model_error_vector = np.concatenate(all_partial_results, axis=0)
        model_error_vector_list.append(model_error_vector)
        vector_matrix_list.append(zwischenvektoren_matrix_bereinigt)
        print("Slices Nummer ", index_number_of_slices, " wurde berechnet")
        # Die Auslagerung der Speicherung auf einen anderen Thread wurde durch Gemini umgesetzt.
        if index_number_of_slices % 10 == 0 or index_number_of_slices == number_of_slices:
            if speicher_thread is not None:
                assert speicher_thread is not None
                if speicher_thread.is_alive():
                    speicher_thread.join()

            model_error_vector_list_copy = model_error_vector_list.copy()
            dot_product_vector_list_copy = dot_product_vector_list.copy()
            vector_matrix_list_copy = vector_matrix_list.copy()

            def save_background(e_list, d_list, m_list, index_slice):
                joblib.dump(e_list, model_error_vector_list_path)
                joblib.dump(d_list, dot_product_vector_list_path)
                joblib.dump(m_list, vector_matrix_list_path)
                print(f" -> Slice {index_slice} erfolgreich im Hintergrund gespeichert.")

            speicher_thread = threading.Thread(
                target=save_background,
                args=(model_error_vector_list_copy, dot_product_vector_list_copy, vector_matrix_list_copy,
                      index_number_of_slices)
            )
            speicher_thread.start()
    joblib.dump(model_error_vector_list, model_error_vector_list_path)
    joblib.dump(dot_product_vector_list, dot_product_vector_list_path)
    joblib.dump(vector_matrix_list, vector_matrix_list_path)


def minimal_box(politicians=None, p=1.0, weighted_by_votes=True,
                start_question=41, end_question=3116, permute_axes=None,
                number_of_slices=1000, number_of_points=10001,
                number_of_repetitions_minimizer=1000, number_of_points_minimizer=10001,
                search_radius_minimizer=1.0, force_optimize_again=False,
                intervall_of_standard_deviation=0.05,
                number_of_standard_deviations=3,
                zero_axes=None):
    if number_of_points % 2 == 0:
        number_of_points += 1
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    if permute_axes is None:
        permute_axes = [False] * 8
    permuted_hash = hashlib.md5(str(permute_axes).encode('utf-8')).hexdigest()
    if zero_axes is None:
        zero_axes = [False] * 8
    else:
        zero_axes = [bool(x) for x in zero_axes]
    zero_hash = f"_{hashlib.md5(str(zero_axes).encode('utf-8')).hexdigest()}" if any(zero_axes) else ""
    slice_appending(politicians=politicians, p=p,
                    weighted_by_votes=weighted_by_votes,
                    start_question=start_question, end_question=end_question, permute_axes=permute_axes,
                    number_of_slices=number_of_slices,
                    number_of_points=number_of_points,
                    number_of_repetitions_minimizer=number_of_repetitions_minimizer,
                    number_of_points_minimizer=number_of_points_minimizer,
                    search_radius_minimizer=search_radius_minimizer,
                    force_optimize_again=force_optimize_again,
                    zero_axes=zero_axes)
    if weighted_by_votes:
        weighted = "weighted"
    else:
        weighted = "not_weighted"
    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    cache_name_minimizer = f"{politicians_hash}_{start_question}_{end_question}_{p}_{weighted}_{permuted_hash}{zero_hash}"
    cache_name_linear_path_slices = f"{cache_name_minimizer}_{number_of_points}"
    cache_folder = Path("cache")
    linear_path_slices_folder = cache_folder / f"linear path slices {cache_name_linear_path_slices}"
    model_error_vector_list_path = linear_path_slices_folder / f"model_error_vector_list"
    if model_error_vector_list_path.exists():
        model_error_vector_list = joblib.load(model_error_vector_list_path)
        if len(model_error_vector_list) >= number_of_slices:
            cache_folder = Path("cache")
            linear_path_slices_folder = cache_folder / f"linear path slices {cache_name_linear_path_slices}"
            linear_path_slices_folder.mkdir(parents=True, exist_ok=True)
            minimizer_folder = cache_folder / f"minimizer {cache_name_minimizer}"
            minimizer_folder.mkdir(parents=True, exist_ok=True)
            arg_min_path = minimizer_folder / f"argmin.pkl"
            minimum_path = minimizer_folder / f"minimum.pkl"
            minimum = joblib.load(minimum_path)
            arg_min = joblib.load(arg_min_path)
            dot_product_vector_list_path = linear_path_slices_folder / f"dot_product_vetcor_list"
            model_error_vector_list_path = linear_path_slices_folder / f"model_error_vector_list"
            vector_matrix_list_path = linear_path_slices_folder / f"vector_matrix_list"
            dot_product_vector_list = joblib.load(dot_product_vector_list_path)
            model_error_vector_list = joblib.load(model_error_vector_list_path)
            vector_matrix_list = joblib.load(vector_matrix_list_path)
            length = len(dot_product_vector_list)
            minimal_matrix = np.full((2 * length, 8), np.nan)
            maximal_matrix = np.full((2 * length, 8), np.nan)
            for index in range(length):
                dot_product_vector = dot_product_vector_list[index]
                intervall_mask_negativ = (dot_product_vector > ((-1) * intervall_of_standard_deviation)) & (
                        dot_product_vector < 0)
                intervall_mask_positiv = (dot_product_vector < intervall_of_standard_deviation) & (
                        dot_product_vector >= 0)
                x_values_negativ = dot_product_vector[intervall_mask_negativ]
                x_values_positiv = dot_product_vector[intervall_mask_positiv]
                vector_matrix = vector_matrix_list[index]
                model_error_vector = model_error_vector_list[index]
                if len(x_values_negativ) > 3:
                    y_values_negativ = model_error_vector[intervall_mask_negativ]
                    trendline_negativ = np.polyfit(x_values_negativ, y_values_negativ, 2)
                    y_trend_negativ = np.polyval(trendline_negativ, x_values_negativ)
                    difference = y_trend_negativ - y_values_negativ
                    standard_deviation = float(np.std(difference))
                    minimum_mask = model_error_vector < minimum + number_of_standard_deviations * standard_deviation
                    vector_matrix_minimum = vector_matrix[minimum_mask]
                    if vector_matrix_minimum.shape[0] != 0:
                        min_vector = np.min(vector_matrix_minimum, axis=0)
                        max_vector = np.max(vector_matrix_minimum, axis=0)
                    else:
                        min_vector = np.maximum(0.0, arg_min)
                        max_vector = np.maximum(0.0, arg_min)
                    minimal_matrix[index] = min_vector
                    maximal_matrix[index] = max_vector
                if len(x_values_positiv) > 3:
                    y_values_positiv = model_error_vector[intervall_mask_positiv]
                    trendline_positiv = np.polyfit(x_values_positiv, y_values_positiv, 2)
                    y_trend_positiv = np.polyval(trendline_positiv, x_values_positiv)
                    difference = y_trend_positiv - y_values_positiv
                    standard_deviation = float(np.std(difference))
                    minimum_mask = model_error_vector < minimum + number_of_standard_deviations * standard_deviation
                    vector_matrix_minimum = vector_matrix[minimum_mask]
                    if vector_matrix_minimum.shape[0] != 0:
                        min_vector = np.min(vector_matrix_minimum, axis=0)
                        max_vector = np.max(vector_matrix_minimum, axis=0)
                    else:
                        min_vector = np.maximum(0.0, arg_min)
                        max_vector = np.maximum(0.0, arg_min)
                    minimal_matrix[index + length] = min_vector
                    maximal_matrix[index+ length] = max_vector
            total_min_vector = np.nanmin(minimal_matrix, axis=0)
            total_max_vector = np.nanmax(maximal_matrix, axis=0)
            return total_min_vector, total_max_vector, arg_min, minimum
    print("Es wurden nicht genügend linear path slices erzeugt")
    return None


def full_results_function(politicians=None, p=1.0, weighted_by_votes=True,
                          start_question=41, end_question=3116, permute_axes=None,
                          number_of_slices=1000, number_of_points=10001,
                          number_of_repetitions_minimizer=1000, number_of_points_minimizer=10001,
                          search_radius_minimizer=1.0, force_optimize_again=False,
                          intervall_of_standard_deviation=0.05,
                          number_of_standard_deviations=3,
                          zero_axes=None):
    if number_of_points % 2 == 0:
        number_of_points += 1
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    if permute_axes is None:
        permute_axes = [False] * 8
    res_min_box = minimal_box(politicians=politicians,
                              p=p,
                              weighted_by_votes=weighted_by_votes,
                              start_question=start_question,
                              end_question=end_question,
                              permute_axes=permute_axes,
                              number_of_slices=number_of_slices,
                              number_of_points=number_of_points,
                              number_of_repetitions_minimizer=number_of_repetitions_minimizer,
                              number_of_points_minimizer=number_of_points_minimizer,
                              search_radius_minimizer=search_radius_minimizer,
                              force_optimize_again=force_optimize_again,
                              intervall_of_standard_deviation=intervall_of_standard_deviation,
                              number_of_standard_deviations=number_of_standard_deviations,
                              zero_axes=zero_axes)
    if res_min_box is None:
        return None
    min_vector, max_vector, argument_min, minimum = res_min_box
    all_politicians = np.arange(203)
    val_block = np.setdiff1d(all_politicians, politicians)
    validation_value_weighted = compute_error_function(weight=argument_min,
                                                       p=p,
                                                       politicians=val_block,
                                                       weighted_by_votes=True,
                                                       start_question=start_question,
                                                       end_question=end_question,
                                                       permute_axes=permute_axes,
                                                       zero_axes=zero_axes)
    validation_value_unweighted = compute_error_function(weight=argument_min, p=p,
                                                         politicians=val_block,
                                                         weighted_by_votes=False,
                                                         start_question=start_question,
                                                         end_question=end_question,
                                                         permute_axes=permute_axes,
                                                         zero_axes=zero_axes)
    validation_baseline_weighted = compute_error_function(weight=np.ones(8), p=1,
                                                          politicians=val_block,
                                                          weighted_by_votes=True,
                                                          start_question=start_question,
                                                          end_question=end_question,
                                                          permute_axes=permute_axes,
                                                          zero_axes=None)
    validation_baseline_unweighted = compute_error_function(weight=np.ones(8), p=1,
                                                            politicians=val_block,
                                                            weighted_by_votes=False,
                                                            start_question=start_question,
                                                            end_question=end_question,
                                                            permute_axes=permute_axes,
                                                            zero_axes=None)
    baseline_weighted = compute_error_function(weight=np.ones(8),
                                               politicians=politicians, p=1,
                                               weighted_by_votes=True,
                                               start_question=start_question,
                                               end_question=end_question,
                                               permute_axes=permute_axes,
                                               zero_axes=None)
    baseline_unweighted = compute_error_function(weight=np.ones(8),
                                                 politicians=politicians, p=1,
                                                 weighted_by_votes=False,
                                                 start_question=start_question,
                                                 end_question=end_question,
                                                 permute_axes=permute_axes,
                                                 zero_axes=None)
    metrik = "Manhattan Metrik"
    if p == 2:
        metrik = "Euklidische Metrik"
    elif p != 1 and p != 2:
        metrik = f"Lp Metrik mit p = {p}"
    gewichtet = "nicht gewichtet"
    if weighted_by_votes:
        gewichtet = "gewichtet"
        minimum_unweighted = compute_error_function(weight=argument_min, politicians=politicians, p=p,
                                                    weighted_by_votes=False,
                                                    start_question=start_question,
                                                    end_question=end_question,
                                                    permute_axes=permute_axes,
                                                    zero_axes=zero_axes)
        minimum_weighted = minimum
    else:
        minimum_unweighted = minimum
        minimum_weighted = compute_error_function(weight=argument_min, politicians=politicians, p=p,
                                                  weighted_by_votes=True,
                                                  start_question=start_question,
                                                  end_question=end_question,
                                                  permute_axes=permute_axes,
                                                  zero_axes=zero_axes)
    print(f"Für {metrik} und {gewichtet}:")
    print("Minimum:      ", format_results(argument_min))
    print("Untergrenzen: ", format_results(min_vector))
    print("Obergrenzen:  ", format_results(max_vector))
    print(f"Die Anzahl Fehler beim Minimum ist {minimum_unweighted.astype(float)} mit einem Gewicht von {minimum_weighted.astype(float)},",
          f"im Vergleich zur Baseline von {baseline_unweighted.astype(float)} Anzahl Fehler mit einem Gewicht von {baseline_weighted.astype(float)}")
    print(
        f"Für die restlichen Politiker ist die Anzahl Fehler bei {validation_value_unweighted.astype(float)} mit Gewicht {validation_value_weighted.astype(float)}",
        f"im Vergleich zur Baseline von {validation_baseline_unweighted.astype(float)} mit Gewicht {validation_baseline_weighted.astype(float)}")
    return True


def plot_linear_slices(politicians=None, p=1.0,
                       weighted_by_votes=True, start_question=41,
                       end_question=3116, permute_axes=None,
                       number_of_slices=1000, number_of_points=10001,
                       number_of_repetitions_minimizer=1000, number_of_points_minimizer=10001,
                       search_radius_minimizer=1.0, force_optimize_again=False,
                       random_lines=False, zero_axes=None):
    if number_of_points % 2 == 0:
        number_of_points += 1
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    if permute_axes is None:
        permute_axes = [False] * 8
    permuted_hash = hashlib.md5(str(permute_axes).encode('utf-8')).hexdigest()
    slice_appending(politicians=politicians, p=p,
                    weighted_by_votes=weighted_by_votes,
                    start_question=start_question, end_question=end_question, permute_axes=permute_axes,
                    number_of_slices=number_of_slices,
                    number_of_points=number_of_points,
                    number_of_repetitions_minimizer=number_of_repetitions_minimizer,
                    number_of_points_minimizer=number_of_points_minimizer,
                    search_radius_minimizer=search_radius_minimizer,
                    force_optimize_again=force_optimize_again,
                    zero_axes=zero_axes)
    if weighted_by_votes:
        weighted = "weighted"
    else:
        weighted = "not_weighted"

    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    if zero_axes is None:
        zero_axes = [False] * 8
    else:
        zero_axes = [bool(x) for x in zero_axes]
    zero_hash = f"{hashlib.md5(str(zero_axes).encode('utf-8')).hexdigest()}_" if any(zero_axes) else ""
    cache_name_linear_path_slices = (
        f"{politicians_hash}_"
        f"{start_question}_"
        f"{end_question}_"
        f"{p}_"
        f"{weighted}_"
        f"{permuted_hash}_"
        f"{zero_hash}"
        f"{number_of_points}"
    )
    cache_folder = Path("cache")
    linear_path_slices_folder = cache_folder / f"linear path slices {cache_name_linear_path_slices}"
    linear_path_slices_folder.mkdir(parents=True, exist_ok=True)
    model_error_vector_list_path = linear_path_slices_folder / f"model_error_vector_list"
    dot_product_vector_list_path = linear_path_slices_folder / f"dot_product_vetcor_list"
    model_error_vector_list = joblib.load(model_error_vector_list_path)
    dot_product_vector_list = joblib.load(dot_product_vector_list_path)
    number_of_lines = len(dot_product_vector_list)
    if number_of_lines < number_of_slices:
        print("Es sind nur ", number_of_lines, "vorhanden")
    if random_lines:
        indices = np.arange(len(dot_product_vector_list))
        np.random.shuffle(indices)
        model_error_vector_array = np.array(model_error_vector_list, dtype=object)
        dot_product_vector_array = np.array(dot_product_vector_list, dtype=object)
        model_error_vector_array_mixed = model_error_vector_array[indices]
        dot_product_vector_array_mixed = dot_product_vector_array[indices]
        model_error_vector_list = model_error_vector_array_mixed.tolist()
        dot_product_vector_list = dot_product_vector_array_mixed.tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    for index in range(min(number_of_lines, number_of_slices)):
        ax.plot(dot_product_vector_list[index], model_error_vector_list[index], linestyle="-", linewidth=1.5, color="darkblue", alpha=0.6)
    ax.set_xlabel(r"Gerichtete Distanz $d(t)$", fontsize=18, labelpad=10)
    ax.set_ylabel(r"Fehlerwert $L$", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


def plot_metric_exponent(politicians=None, weighted_by_votes=True,
                         start_question=41, end_question=3116, permute_axes=None,
                         number_of_repetitions = 1000, number_of_points = 10001,
                         search_radius = 1.0, force_optimize_again = False,
                         start_exponent = 1.0, end_exponent = 3.0, number_of_exponents = 10, zero_axes=None):
    if number_of_points % 2 == 0:
        number_of_points += 1
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    if permute_axes is None:
        permute_axes = [False] * 8
    permuted_hash = hashlib.md5(str(permute_axes).encode('utf-8')).hexdigest()
    if weighted_by_votes:
        weighted = "weighted"
    else:
        weighted = "not_weighted"
    politicians_hash = hashlib.md5(str(politicians).encode('utf-8')).hexdigest()
    if zero_axes is None:
        zero_axes = [False] * 8
    else:
        zero_axes = [bool(x) for x in zero_axes]
    zero_hash = f"_{hashlib.md5(str(zero_axes).encode('utf-8')).hexdigest()}" if any(zero_axes) else ""
    cache_folder = Path("cache")
    exponent_list = []
    value_list = []
    for exponent in np.linspace(start_exponent, end_exponent, number_of_exponents):
        exponent = round(float(exponent), 3)
        minimizer(politicians=politicians, p=exponent, weighted_by_votes=weighted_by_votes,
                  start_question=start_question, end_question=end_question,
                  number_of_repetitions=number_of_repetitions, number_of_points=number_of_points,
                  search_radius=search_radius, force_optimize_again=force_optimize_again, zero_axes=zero_axes)
        cache_name_minimizer = f"{politicians_hash}_{start_question}_{end_question}_{exponent}_{weighted}_{permuted_hash}{zero_hash}"
        minimizer_folder = cache_folder / f"minimizer {cache_name_minimizer}"
        minimum_path = minimizer_folder / f"minimum.pkl"
        value = joblib.load(minimum_path)
        value_list.append(value)
        exponent_list.append(exponent)
    best_index = np.argmin(value_list)
    best_exponent = exponent_list[best_index]
    best_value = value_list[best_index]
    # Die Erstellung des Graphen wurde mithilfe von ChatGPT umgesetzt.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(best_exponent, best_value, s=60, zorder=5, color="darkblue")
    ax.annotate(rf"$p={best_exponent:.2f}$", xy=(best_exponent, best_value), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=14)
    ax.plot(exponent_list, value_list, linestyle="-", linewidth=2,markersize=6, color="darkblue")
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel(r"Exponent $p$", fontsize=18, labelpad=10)
    ax.set_ylabel(r"Fehlerwert $L^*(p)$", fontsize=18, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
    best_exponent = exponent_list[np.argmin(value_list)]
    return best_exponent


def weighted_distance(politician_a, politician_b, weight, p):
    distance = np.sum(weight * (np.abs(politician_a - politician_b) ** p))
    return distance

# Die genauen Farbcodes wurden von ChatGPT geliefert.
party_color_dictionary = {
    1:  "#FF8200",   # Die Mitte
    2:  "#0088CE",   # FDP
    3:  "#006D2C",   # SVP
    4:  "#F02311",   # SP
    5: "#66A61E",   # Grüne
    6: "#C7B700",   # GLP
    7:  "#8C564B",   # EDU
    8:  "#E69F00",   # MCG
    9: "#00A6A6",  # EVP
    10: "#7B4F9E"   # LEGA
    }

# Die Funktion zur Erstellung der Clusteranalyse wurde mithilfe von Gemini erstellt.

def plot_cluster_analysis(weight=np.ones(8), p=1.0, politicians=None):
    if politicians is None:
        politicians = list(range(203))
    politicians_matrix = politicians_groups_and_matrix(politicians=politicians)[0]
    excel_tab = pd.read_excel(
        r'C:\Users\konra\OneDrive - DIG-IT - eduBS\Dokumente\Gymnasium\MATURAARBEIT\Datei fuer Python.xlsx',
        header=None)
    politicians_slice = excel_tab.iloc[9, 12:215].copy()
    politicians_slice_numeric = pd.to_numeric(politicians_slice, errors='coerce')
    politicians_party_vector: np.ndarray = np.array(politicians_slice_numeric, dtype=np.float64)
    party_vector = politicians_party_vector[politicians]
    politicians_matrix_transposed = politicians_matrix.T
    distance_matrix_raw = scipy.spatial.distance.pdist(politicians_matrix_transposed, metric=lambda politician_a, politician_b: weighted_distance(politician_a, politician_b, weight, p))
    min_value = np.min(distance_matrix_raw)
    distance_matrix = distance_matrix_raw - min_value
    cluster_data = scipy.cluster.hierarchy.linkage(distance_matrix, method="average")
    def get_color_func(cluster_d):
        root_node = scipy.cluster.hierarchy.to_tree(cluster_d)
        node_dict = {}
        def build_node_dict(node):
            if node is None:
                return
            node_dict[node.id] = node
            build_node_dict(node.left)
            build_node_dict(node.right)
        build_node_dict(root_node)
        def color_func(node_id):
            if node_id in node_dict:
                target_node = node_dict[node_id]
                if target_node.is_leaf():
                    politician_idx = target_node.id
                    party_id = int(party_vector[politician_idx])
                    return party_color_dictionary.get(party_id, "black")
                leaf_indices = target_node.pre_order(lambda x: x.id)
                parties_in_branch = set(party_vector[i] for i in leaf_indices)
                if len(parties_in_branch) == 1:
                    party_id = int(parties_in_branch.pop())
                    return party_color_dictionary.get(party_id, "black")
            return "black"
        return color_func
    fig, ax = plt.subplots( figsize=(10, 6))
    scipy.cluster.hierarchy.dendrogram(cluster_data, ax=ax, link_color_func=get_color_func(cluster_data))
    ddata = scipy.cluster.hierarchy.dendrogram(
        cluster_data,
        ax=ax,
        link_color_func=get_color_func(cluster_data),
        no_labels=True
    )
    leaves_order = ddata['leaves'] or []
    party_colors_leaves = [party_color_dictionary.get(int(party_vector[i]), 'black') for i in leaves_order]
    ax.scatter(
        np.arange(5, len(leaves_order) * 10, 10),
        [0] * len(leaves_order),
        color=party_colors_leaves,
        s=10, zorder=10
    )
    plt.tight_layout()
    ax.axis("off")
    plt.show()


# Die Funktion zur Erstellung der Legende wurde vollständig durch ChatGPT erstellt.

def plot_party_legend():
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=party_color_dictionary[1], label="Die Mitte"),
        Patch(facecolor=party_color_dictionary[2], label="FDP"),
        Patch(facecolor=party_color_dictionary[3], label="SVP"),
        Patch(facecolor=party_color_dictionary[4], label="SP"),
        Patch(facecolor=party_color_dictionary[5], label="Grüne"),
        Patch(facecolor=party_color_dictionary[6], label="GLP"),
        Patch(facecolor=party_color_dictionary[7], label="EDU"),
        Patch(facecolor=party_color_dictionary[8], label="MCG"),
        Patch(facecolor=party_color_dictionary[9], label="EVP"),
        Patch(facecolor=party_color_dictionary[10], label="LEGA")
    ]
    fig = plt.figure(figsize=(10, 1))
    fig.legend(handles=legend_elements, loc="center", ncol=5, frameon=False, fontsize=14)
    plt.show()



def compute_weighted_distance_matrix(weight=np.ones(8), politicians=None, p=1.0):
    full_politicians_matrix = politicians_groups_and_matrix(politicians=None)[0]
    full_politicians_difference_tensor = full_politicians_matrix.T[np.newaxis, : , :] - full_politicians_matrix.T[:, np.newaxis, :]
    full_politicians_difference_tensor = np.abs(full_politicians_difference_tensor)
    politicians_difference_tensor = full_politicians_difference_tensor[politicians, :, :]
    politicians_distance_tensor = politicians_difference_tensor ** p
    weighted_powered_lp_distance_tensor = politicians_distance_tensor * weight[np.newaxis, np.newaxis, :]
    weighted_powered_lp_distance_matrix = np.sum(weighted_powered_lp_distance_tensor, axis=2)
    return weighted_powered_lp_distance_matrix


def finding_impressive_examples(politicians_for_example=None, politicians_for_minimizer=None, p=1.0,
                                                 weighted_by_votes=True, start_question=41,
                                                 end_question=3116, permute_axes=None,
                                                 number_of_repetitions_minimizer=300, number_of_points_minimizer=10001,
                                                 search_radius_minimizer=1.0, force_optimize_again=False, number_of_best_examples=10, zero_axes=None):
    if politicians_for_example is None:
        politicians_for_example = list(range(203))
    politicians_for_example = list(sorted(politicians_for_example))
    if politicians_for_minimizer is None:
        politicians_for_minimizer = list(range(203))
    politicians_for_minimizer = list(sorted(politicians_for_minimizer))
    full_shared_vote_ratio_matrix = compute_shared_vote_ratio_matrix(politicians=None, start_question=start_question, end_question=end_question)
    politicians_shared_vote_ratio_matrix = full_shared_vote_ratio_matrix[politicians_for_example, :]
    shared_vote_ratio_a_b_tensor = politicians_shared_vote_ratio_matrix[:, :, np.newaxis]
    shared_vote_ratio_a_c_tensor = politicians_shared_vote_ratio_matrix[:, np.newaxis, :]
    shared_vote_ratio_difference_tensor = shared_vote_ratio_a_b_tensor - shared_vote_ratio_a_c_tensor
    weight = minimizer(politicians=politicians_for_minimizer, p=p,
                       weighted_by_votes=weighted_by_votes,
                       start_question=start_question, end_question=end_question,
                       permute_axes=permute_axes,
                       number_of_repetitions=number_of_repetitions_minimizer,
                       number_of_points=number_of_points_minimizer,
                       search_radius=search_radius_minimizer,
                       force_optimize_again=force_optimize_again, zero_axes=zero_axes)
    print(weight)
    weighted_powered_lp_difference_matrix = compute_weighted_distance_matrix(weight=weight, politicians=politicians_for_example, p=p)
    weighted_powered_lp_difference_a_b_tensor = weighted_powered_lp_difference_matrix[:, :, np.newaxis]
    weighted_powered_lp_difference_a_c_tensor = weighted_powered_lp_difference_matrix[:, np.newaxis, :]
    weighted_powered_lp_difference_a_b_c_tensor = weighted_powered_lp_difference_a_c_tensor - weighted_powered_lp_difference_a_b_tensor
    unweighted_powered_lp_difference_matrix = compute_weighted_distance_matrix(weight=np.ones(8), politicians=politicians_for_example, p=1.0)
    unweighted_powered_lp_difference_a_b_tensor = unweighted_powered_lp_difference_matrix[:, :, np.newaxis]
    unweighted_powered_lp_difference_a_c_tensor = unweighted_powered_lp_difference_matrix[:, np.newaxis, :]
    unweighted_powered_lp_difference_a_b_c_tensor = unweighted_powered_lp_difference_a_c_tensor - unweighted_powered_lp_difference_a_b_tensor
    index_a_list = []
    index_b_list = []
    index_c_list = []
    shared_vote_ratio_difference_list = []
    print(shared_vote_ratio_difference_tensor.shape, weighted_powered_lp_difference_a_b_c_tensor.shape)
    for index_a in range(len(politicians_for_example)):
        for index_b in range(203):
            for index_c in range(index_b, 203):
                if shared_vote_ratio_difference_tensor[index_a, index_b, index_c] > 0:
                    if weighted_powered_lp_difference_a_b_c_tensor[index_a, index_b, index_c] > 0:
                        if unweighted_powered_lp_difference_a_b_c_tensor[index_a, index_b, index_c] < 0:
                            index_a_list.append(index_a)
                            index_b_list.append(index_b)
                            index_c_list.append(index_c)
                            shared_vote_ratio_difference_list.append(abs(shared_vote_ratio_difference_tensor[index_a, index_b, index_c]))
                if shared_vote_ratio_difference_tensor[index_a, index_b, index_c] < 0:
                    if weighted_powered_lp_difference_a_b_c_tensor[index_a, index_b, index_c] < 0:
                        if unweighted_powered_lp_difference_a_b_c_tensor[index_a, index_b, index_c] > 0:
                            index_a_list.append(index_a)
                            index_b_list.append(index_b)
                            index_c_list.append(index_c)
                            shared_vote_ratio_difference_list.append(abs(shared_vote_ratio_difference_tensor[index_a, index_b, index_c]))
    shared_vote_ratio_difference_list_array = np.array(shared_vote_ratio_difference_list)
    index_a_list_array = np.array(index_a_list)
    index_b_list_array = np.array(index_b_list)
    index_c_list_array = np.array(index_c_list)
    sorting_indices = np.argsort(shared_vote_ratio_difference_list)[::-1]
    sorted_shared_vote_ratio_difference_list = shared_vote_ratio_difference_list_array[sorting_indices]
    sorted_index_a_list = index_a_list_array[sorting_indices]
    sorted_index_b_list = index_b_list_array[sorting_indices]
    sorted_index_c_list = index_c_list_array[sorting_indices]
    full_politicians_matrix = politicians_groups_and_matrix(politicians=None)[0]
    politicians_matrix = full_politicians_matrix[:, politicians_for_example]
    for index in range(number_of_best_examples):
        print(full_shared_vote_ratio_matrix[150+sorted_index_a_list[index], sorted_index_b_list[index]])
        print(full_shared_vote_ratio_matrix[150 + sorted_index_a_list[index], sorted_index_c_list[index]])
        print("weighted lp distance", weighted_powered_lp_difference_a_b_c_tensor[sorted_index_a_list[index], sorted_index_b_list[index], sorted_index_c_list[index]])
        print("unweighted manhattan distance", unweighted_powered_lp_difference_a_b_c_tensor[sorted_index_a_list[index], sorted_index_b_list[index],
                                                          sorted_index_c_list[index]])

        print(politicians_matrix[:, sorted_index_a_list[index]])
        print(full_politicians_matrix[:, sorted_index_a_list[index] + 150])
        print(full_politicians_matrix[:, sorted_index_b_list[index]])
        print(full_politicians_matrix[:, sorted_index_c_list[index]])
        print(sorted_index_a_list[index], sorted_index_b_list[index], sorted_index_c_list[index], sorted_shared_vote_ratio_difference_list[index])

def compute_correlation_index(politicians=None):
    full_politicians_matrix = politicians_groups_and_matrix()[0]
    politicians_matrix = full_politicians_matrix[:, politicians]
    r_matrix = np.corrcoef(politicians_matrix)
    return r_matrix

# Die Funktion zur Erstellung des gewichteten Smartspider wurde anhand einer Skizze von ChatGPT erstellt.

def plot_weighted_net_diagram(politicians=None, p=1.0,
                              weighted_by_votes=True, start_question=41,
                              end_question=3116, permute_axes=None,
                              number_of_repetitions_minimizer=300, number_of_points_minimizer=10001,
                              search_radius_minimizer=1.0, force_optimize_again=False, zero_axes=None, values_1=None, values_2=None):
    zero_axes_array = np.array(zero_axes, dtype=bool)
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    weight = minimizer(politicians=politicians, p=p,
                       weighted_by_votes=weighted_by_votes,
                       start_question=start_question, end_question=end_question,
                       permute_axes=permute_axes,
                       number_of_repetitions=number_of_repetitions_minimizer,
                       number_of_points=number_of_points_minimizer,
                       search_radius=search_radius_minimizer,
                       force_optimize_again=force_optimize_again, zero_axes=zero_axes)
    if values_1 is None:
        values_1 = [0.25] * 5
    if values_2 is None:
        values_2 = [0.75] * 5
    name_1 = "Nutzer"
    name_2 = "Kandidierender"
    weights = weight[~zero_axes_array]
    labels = [
        "Offene Aussenpolitik",
        "Liberale Wirtschaft",
        "Law & Order",
        "Restriktive Migration",
        "Ausgebauter Umweltschutz"
    ]
    figsize = (9, 7)
    weights = np.asarray(weights, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)
    weights = np.asarray(weights, dtype=float)
    angles = np.deg2rad([
        90,  # Aussenpolitik
        18,  # Wirtschaft
        -54,  # Law & Order
        -126,  # Migration
        162  # Umweltschutz
    ])
    axis_lengths = weights ** (1 / p)
    axis_lengths /= axis_lengths.max()
    axis_x = axis_lengths * np.cos(angles)
    axis_y = axis_lengths * np.sin(angles)
    def draw_person(values, color="blue", linestyle="-", linewidth=2):
        point_lengths = values * axis_lengths

        point_x = point_lengths * np.cos(angles)
        point_y = point_lengths * np.sin(angles)

        # Polygon schliessen
        polygon_x = np.append(point_x, point_x[0])
        polygon_y = np.append(point_y, point_y[0])

        # Profil
        ax.plot(
            polygon_x,
            polygon_y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )

        # Füllung
        ax.fill(
            polygon_x,
            polygon_y,
            color=color,
            alpha=0.08
        )

        # Punkte der Person
        ax.scatter(
            point_x,
            point_y,
            color=color,
            s=45,
            zorder=5
        )
    fig, ax = plt.subplots(figsize=figsize)
    for x, y in zip(axis_x, axis_y):
        ax.plot(
            [0, x],
            [0, y],
            color="black",
            linewidth=2,
            zorder=1
        )
    endstrich_laenge = 0.03
    for angle, length in zip(angles, axis_lengths):
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        nx = -np.sin(angle)
        ny = np.cos(angle)
        ax.plot(
            [
                x - endstrich_laenge * nx,
                x + endstrich_laenge * nx
            ],
            [
                y - endstrich_laenge * ny,
                y + endstrich_laenge * ny
            ],
            color="black",
            linewidth=2,
            zorder=3
        )
    ax.scatter(
        [0],
        [0],
        color="black",
        s=20,
        zorder=2
    )
    draw_person(
        values_1,
        color="blue",
        linestyle="-"
    )
    draw_person(
        values_2,
        color="red",
        linestyle="-"
    )
    label_offsets = [
        (0.00, 0.10),  # Aussenpolitik
        (0.06, 0.03),  # Wirtschaft
        (0.05, -0.01),  # Law & Order
        (-0.05, -0.02),  # Migration
        (-0.05, 0.04)  # Umwelt
    ]
    for i, (angle, length, label) in enumerate(
            zip(angles, axis_lengths, labels)
    ):

        x = length * np.cos(angle)
        y = length * np.sin(angle)

        dx, dy = label_offsets[i]

        if x > 0.15:
            ha = "left"
        elif x < -0.15:
            ha = "right"
        else:
            ha = "center"

        if y > 0.15:
            va = "bottom"
        elif y < -0.15:
            va = "top"
        else:
            va = "center"

        ax.text(
            x + dx,
            y + dy,
            label,
            ha=ha,
            va=va,
            fontsize=16
        )
    ax.plot(
        [], [],
        color="blue",
        linestyle="-",
        linewidth=2,
        label=name_1
    )

    ax.plot(
        [], [],
        color="red",
        linestyle="-",
        linewidth=2,
        label=name_2
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.1),
        fontsize=16
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# Die Funktion zur Erstellung des gewichteten Balkendiagramms wurde anhand einer Skizze von ChatGPT erstellt.

def plot_weighted_bar_diagram(politicians=None, p=1.0,
                              weighted_by_votes=True, start_question=41,
                              end_question=3116, permute_axes=None,
                              number_of_repetitions_minimizer=300, number_of_points_minimizer=10001,
                              search_radius_minimizer=1.0, force_optimize_again=False, zero_axes=None, values_1=None, values_2=None):
    zero_axes_array = np.array(zero_axes, dtype=bool)
    if politicians is None:
        politicians = list(range(203))
    politicians = list(sorted(politicians))
    weight = minimizer(politicians=politicians, p=p,
                       weighted_by_votes=weighted_by_votes,
                       start_question=start_question, end_question=end_question,
                       permute_axes=permute_axes,
                       number_of_repetitions=number_of_repetitions_minimizer,
                       number_of_points=number_of_points_minimizer,
                       search_radius=search_radius_minimizer,
                       force_optimize_again=force_optimize_again, zero_axes=zero_axes)
    if values_1 is None:
        values_1 = [0.25] * 5
    if values_2 is None:
        values_2 = [0.75] * 5
    name_1 = "Nutzer"
    name_2 = "Kandidierender"
    weights = weight[~zero_axes_array]
    #Hier wird CHATGPT verwendet
    labels = [
        "Offene Aussenpolitik",
        "Liberale Wirtschaft",
        "Law & Order",
        "Restriktive Migration",
        "Ausgebauter Umweltschutz"
    ]
    values_1 = np.asarray(values_1, dtype=float)
    weights = np.asarray(weights, dtype=float)
    axis_lengths = weights ** (1 / p)
    axis_lengths /= axis_lengths.max()
    axis_lengths *= 1.3
    position_1 = values_1 * axis_lengths + 0.1
    position_2 = values_2 * axis_lengths + 0.1
    optimized_distance = np.sum(weights * np.abs(values_1 - values_2) ** p)
    distance_max = 8.0
    if optimized_distance > distance_max:
        print(
            f"Warnung: Die berechnete Distanz "
            f"{float(optimized_distance):.3f} liegt über der "
            f"Darstellungsskala von 0 bis {distance_max:.0f}."
        )
    distance_position = optimized_distance / distance_max
    distance_position = distance_position * 1.3 + 0.1
    fig, ax = plt.subplots(figsize=(14, 9))
    y_positions = np.arange(5)
    tick_values = [0, 25, 50, 75, 100]

    for i, (y, length) in enumerate(
            zip(y_positions, axis_lengths)
    ):
        ax.plot(
            [0.1, length + 0.1],
            [y, y],
            color="black",
            linewidth=5,
            solid_capstyle="butt",
            zorder=1
        )
        for tick in tick_values:
            relative_position = tick / 100
            x = relative_position * length + 0.1
            tick_height = 0.1
            ax.plot(
                [x, x],
                [
                    y - tick_height,
                    y + tick_height
                ],
                color="black",
                linewidth=1,
                zorder=2
            )
            ax.text(
                x,
                y - 0.32,
                str(tick),
                ha="center",
                va="top",
                fontsize=16
            )
        ax.text(
            -0.05,
            y,
            rf"$w_{i+1}={weights[i]:.2f}$",
            va="center",
            ha="right",
            fontsize=16
        )

    ax.scatter(
        position_1,
        y_positions,
        color="blue",
        s=75,
        zorder=5,
        label=name_1
    )
    ax.scatter(
        position_2,
        y_positions,
        color="red",
        s=75,
        zorder=5,
        label=name_2
    )
    ax.legend(loc="upper right", fontsize=16)

    ax.set_yticks([])
    label_x = -1.1
    for i, label in enumerate(labels):
        ax.text(
            label_x,
            y_positions[i],
            label,
            ha="left",
            va="center",
            fontsize=18
        )
    distance_y = 5.5
    ax.plot(
        [0.1, 1.3 + 0.1],
        [distance_y, distance_y],
        color="black",
        linewidth=5,
        solid_capstyle="butt",
        zorder=1
    )
    for tick in [0, 4, 8]:
        relative_position = tick / distance_max
        relative_position = relative_position * 1.3 + 0.1
        ax.plot(
            [relative_position, relative_position],
            [distance_y - 0.1, distance_y + 0.1],
            color="black",
            linewidth=1,
            zorder=2
        )
        ax.text(
            relative_position,
            distance_y - 0.32,
            str(tick),
            ha="center",
            va="top",
            fontsize=16
        )
    ax.scatter(
        [distance_position],
        [distance_y],
        color="black",
        s=75,
        zorder=5
    )
    ax.text(
        label_x,
        distance_y,
        "Optimierte Distanz",
        ha="left",
        va="center",
        fontsize=18
    )
    ax.text(
        distance_position,
        distance_y + 0.45,
        f"{float(optimized_distance):.3f}",
        ha="center",
        va="bottom",
        fontsize=18
    )
    ax.set_xlim(
        -1.20,
        1.55
    )
    ax.set_ylim(
        6.05,
        -0.5
    )
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.show()

# Die Funktion zur Erstellung der MDS-Karte wurde durch ChatGPT erstellt.

def plot_mds_diagram(politicians=None, p=1.0, weighted_by_votes=True, start_question=41,
                     end_question=3116, permute_axes=None,
                     number_of_repetitions_minimizer=300, number_of_points_minimizer=10001,
                     search_radius_minimizer=1.0, force_optimize_again=False, zero_axes=None, values=None):
    if values is None:
        values = (-1.5, 0.5)
    weight = minimizer(politicians=politicians, p=p,
                       weighted_by_votes=weighted_by_votes,
                       start_question=start_question, end_question=end_question,
                       permute_axes=permute_axes,
                       number_of_repetitions=number_of_repetitions_minimizer,
                       number_of_points=number_of_points_minimizer,
                       search_radius=search_radius_minimizer,
                       force_optimize_again=force_optimize_again, zero_axes=zero_axes)
    excel_tab = pd.read_excel(
        r'C:\Users\konra\OneDrive - DIG-IT - eduBS\Dokumente\Gymnasium\MATURAARBEIT\Datei fuer Python.xlsx',
        header=None)
    politicians_slice = excel_tab.iloc[9, 12:215].copy()
    politicians_slice_numeric = pd.to_numeric(politicians_slice, errors='coerce')
    politicians_party_vector: np.ndarray = np.array(politicians_slice_numeric, dtype=np.float64)
    party_vector = politicians_party_vector[list(range(203))]
    party_name = ["Rest", "Mitte", "FDP", "SVP", "SP", "Grüne", "GLP", "EDU", "MCG", "EVP", "LEGA"]
    distance_matrix = compute_weighted_distance_matrix(weight=weight, politicians=list(range(203)), p = p)
    #ChatGPT
    mds = MDS(
        n_components=2,
        metric_mds=True,
        metric="precomputed",
        n_init=20,
        max_iter=1000,
        eps=1e-6,
        random_state=42,
        normalized_stress=True,
        init="random"
    )
    coordinates = mds.fit_transform(distance_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_parties = np.unique(party_vector)
    ax.scatter(values[0], values[1], s=75, color="black", label="Nutzer")
    for party in unique_parties:
        mask = party_vector == party
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=party_name[int(party)],
            color = party_color_dictionary[int(party)],
            s=40
        )
    ax.tick_params(
        axis="both",
        labelsize=16
    )
    ax.set_xlabel("MDS-Dimension 1", fontsize=20)
    ax.set_ylabel("MDS-Dimension 2", fontsize=20)
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=20
    )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    all_indices = np.arange(203)
    block_a_indices = all_indices[:150]
    full_results_function(politicians=block_a_indices, p=1.2)
    for axis in range(8):
        permute_axis_mask_True = np.array([i == axis for i in range(8)])
        print(permute_axis_mask_True)
        permute_axis_mask_False = np.array([i != axis for i in range(8)])
        print(minimizer(politicians=block_a_indices, p=1.2, permute_axes=permute_axis_mask_True))
        print(minimizer(politicians=block_a_indices, p=1.2, permute_axes=permute_axis_mask_False))
    for axis in range(8):
        zero_axis_mask = np.array([i == axis for i in range(8)])
        print(minimizer(politicians=block_a_indices, p=1.2, zero_axes=zero_axis_mask))
    wirtschaft_cluster_maske = [i in [1, 2, 5, 6] for i in range(8)]
    law_cluster_maske = [i in [3] for i in range(8)]
    migrations_aussenpolitik_cluster_maske = [i in [0, 4] for i in range(8)]
    print(minimizer(politicians=block_a_indices, p=1.2, zero_axes=wirtschaft_cluster_maske))
    print(minimizer(politicians=block_a_indices, p=1.2, zero_axes=law_cluster_maske))
    print(minimizer(politicians=block_a_indices, p=1.2, zero_axes=migrations_aussenpolitik_cluster_maske))
    print(compute_correlation_index(block_a_indices))
    zero_axes_mask = [i in [2, 6, 7] for i in range(8)]
    full_results_function(block_a_indices, p=1.2, zero_axes=zero_axes_mask)
    plot_metric_exponent(block_a_indices, start_exponent=0.5, end_exponent=2.5, number_of_exponents=41)
    plot_linear_slices(politicians=block_a_indices, p=1.2)
    opt_weight = minimizer(politicians=block_a_indices, p =1.2, zero_axes=zero_axes_mask)
    plot_cluster_analysis(weight=opt_weight, p=1.2)
    plot_cluster_analysis()
    plot_party_legend()
    plot_weighted_net_diagram(politicians=block_a_indices, p=1.2, zero_axes=zero_axes_mask)
    plot_weighted_bar_diagram(politicians=block_a_indices, p=1.2, zero_axes=zero_axes_mask)
    plot_mds_diagram(politicians=block_a_indices, p=1.2)

