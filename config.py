from pathlib import Path
import os

current_file_directory = Path(__file__).parent

# cedar dataset
class Cedar_config():
    cedar_preprocessing_dir_path = Path(current_file_directory / "output" / "cedar" / "preprocessed_signatures")
    os.makedirs(cedar_preprocessing_dir_path, exist_ok=True)

    cedar_signature_path = Path(current_file_directory / "resources" / "datasets" / "cedar_signatures")
    cedar_forge_dir = Path(cedar_signature_path, 'full_forg')
    cedar_real_dir = Path(cedar_signature_path, 'full_org')

    cedar_output_forge_dir = os.path.join(cedar_preprocessing_dir_path, 'preprocessed_forge')
    cedar_output_real_dir = os.path.join(cedar_preprocessing_dir_path, 'preprocessed_real')

    os.makedirs(cedar_output_forge_dir, exist_ok=True)
    os.makedirs(cedar_output_real_dir, exist_ok=True)


# test_dataset
class test_dataset_config():
    dataset_name = "dataset_2_signatures"
    test_dataset_2_preprocessing_dir_path = Path(current_file_directory / "output" / dataset_name / "preprocessed_signatures")
    os.makedirs(test_dataset_2_preprocessing_dir_path, exist_ok=True)

    test_dataset_2_signature_path = Path(current_file_directory / "resources" / "datasets" / dataset_name)
    test_dataset_2_forge_dir = Path(test_dataset_2_signature_path, 'forg')
    test_dataset_2_real_dir = Path(test_dataset_2_signature_path, 'real')

    test_dataset_2_output_forge_dir = os.path.join(test_dataset_2_preprocessing_dir_path, 'preprocessed_forge')
    test_dataset_2_output_real_dir = os.path.join(test_dataset_2_preprocessing_dir_path, 'preprocessed_real')

    os.makedirs(test_dataset_2_output_forge_dir, exist_ok=True)
    os.makedirs(test_dataset_2_output_real_dir, exist_ok=True)