#!/usr/bin/env python
"""
Dataset Preparation Script

This script downloads and preprocesses medical image datasets for the Fair LLM Medical Diagnosis project.
It handles the CheXRay, Pathology, and Retinal image datasets.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
import random
from PIL import Image
import pydicom


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare medical image datasets")
    
    parser.add_argument("--dataset", type=str, required=True, choices=["chexray", "pathology", "retinal", "all"],
                        help="Dataset to prepare")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save datasets")
    parser.add_argument("--download", action="store_true", help="Whether to download the datasets")
    parser.add_argument("--preprocess", action="store_true", help="Whether to preprocess the datasets")
    parser.add_argument("--split", action="store_true", help="Whether to split the datasets into train/val/test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def download_file(url, destination):
    """Download a file from a URL to a destination."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    
    with open(destination, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=os.path.basename(destination)
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)


def extract_archive(archive_path, extract_dir):
    """Extract an archive to a directory."""
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {archive_path} to {extract_dir}")
    
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def prepare_chexray_dataset(args):
    """Prepare the CheXRay dataset."""
    dataset_dir = os.path.join(args.output_dir, "chexray")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download dataset
    if args.download:
        print("Downloading CheXRay dataset...")
        # Note: In a real implementation, you would download from the actual source
        # For this example, we'll simulate the download
        print("CheXRay dataset download simulation complete.")
    
    # Preprocess dataset
    if args.preprocess:
        print("Preprocessing CheXRay dataset...")
        
        # Create metadata
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            # In a real implementation, you would process actual image files
            # For this example, we'll create a simulated metadata file
            
            # Create simulated metadata
            num_samples = 1000
            demographics = ["African_American", "Asian", "Caucasian", "Hispanic", "Other"]
            
            # Create paths and labels
            paths = [f"images/img_{i:04d}.png" for i in range(num_samples)]
            pneumonia_labels = np.random.randint(0, 2, size=num_samples)
            effusion_labels = np.random.randint(0, 2, size=num_samples)
            atelectasis_labels = np.random.randint(0, 2, size=num_samples)
            
            # Assign demographics with bias
            # Simulate a bias where certain demographics have higher rates of positive labels
            demographic_assignments = []
            for i in range(num_samples):
                if i < num_samples * 0.2:
                    demographic = demographics[0]  # African_American
                    # Increase pneumonia rate for this demographic to simulate bias
                    if pneumonia_labels[i] == 0 and random.random() < 0.3:
                        pneumonia_labels[i] = 1
                elif i < num_samples * 0.4:
                    demographic = demographics[1]  # Asian
                elif i < num_samples * 0.7:
                    demographic = demographics[2]  # Caucasian
                    # Decrease pneumonia rate for this demographic to simulate bias
                    if pneumonia_labels[i] == 1 and random.random() < 0.3:
                        pneumonia_labels[i] = 0
                elif i < num_samples * 0.9:
                    demographic = demographics[3]  # Hispanic
                else:
                    demographic = demographics[4]  # Other
                
                demographic_assignments.append(demographic)
            
            # Create splits
            splits = []
            for i in range(num_samples):
                if i < num_samples * 0.7:
                    splits.append("train")
                elif i < num_samples * 0.85:
                    splits.append("val")
                else:
                    splits.append("test")
            
            # Create DataFrame
            metadata = pd.DataFrame({
                "path": paths,
                "Pneumonia": pneumonia_labels,
                "Effusion": effusion_labels,
                "Atelectasis": atelectasis_labels,
                "demographic": demographic_assignments,
                "split": splits,
            })
            
            # Save metadata
            metadata.to_csv(metadata_path, index=False)
            
            print(f"Created simulated metadata: {metadata_path}")
        else:
            print(f"Metadata already exists: {metadata_path}")
    
    # Split dataset
    if args.split:
        print("Splitting CheXRay dataset...")
        
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        if os.path.exists(metadata_path):
            # Load metadata
            metadata = pd.read_csv(metadata_path)
            
            # Check if split column exists
            if "split" not in metadata.columns:
                print("Creating train/val/test splits...")
                
                # Set random seed
                random.seed(args.seed)
                np.random.seed(args.seed)
                
                # Create splits
                num_samples = len(metadata)
                indices = np.random.permutation(num_samples)
                
                train_end = int(num_samples * 0.7)
                val_end = int(num_samples * 0.85)
                
                train_indices = indices[:train_end]
                val_indices = indices[train_end:val_end]
                test_indices = indices[val_end:]
                
                # Assign splits
                splits = [""] * num_samples
                for i in train_indices:
                    splits[i] = "train"
                for i in val_indices:
                    splits[i] = "val"
                for i in test_indices:
                    splits[i] = "test"
                
                # Add split column
                metadata["split"] = splits
                
                # Save metadata
                metadata.to_csv(metadata_path, index=False)
                
                print("Splits created and saved.")
            else:
                print("Splits already exist in metadata.")
        else:
            print(f"Metadata file not found: {metadata_path}")
    
    print("CheXRay dataset preparation complete.")


def prepare_pathology_dataset(args):
    """Prepare the Pathology dataset."""
    dataset_dir = os.path.join(args.output_dir, "pathology")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download dataset
    if args.download:
        print("Downloading Pathology dataset...")
        # Note: In a real implementation, you would download from the actual source
        # For this example, we'll simulate the download
        print("Pathology dataset download simulation complete.")
    
    # Preprocess dataset
    if args.preprocess:
        print("Preprocessing Pathology dataset...")
        
        # Create metadata
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            # In a real implementation, you would process actual image files
            # For this example, we'll create a simulated metadata file
            
            # Create simulated metadata
            num_samples = 800
            demographics = ["African_American", "Asian", "Caucasian", "Hispanic", "Other"]
            
            # Create paths and labels
            paths = [f"images/slide_{i:04d}.png" for i in range(num_samples)]
            malignant_labels = np.random.randint(0, 2, size=num_samples)
            
            # Assign demographics with bias
            # Simulate a bias where certain demographics have higher rates of positive labels
            demographic_assignments = []
            for i in range(num_samples):
                if i < num_samples * 0.15:
                    demographic = demographics[0]  # African_American
                    # Increase malignancy rate for this demographic to simulate bias
                    if malignant_labels[i] == 0 and random.random() < 0.4:
                        malignant_labels[i] = 1
                elif i < num_samples * 0.35:
                    demographic = demographics[1]  # Asian
                elif i < num_samples * 0.65:
                    demographic = demographics[2]  # Caucasian
                    # Decrease malignancy rate for this demographic to simulate bias
                    if malignant_labels[i] == 1 and random.random() < 0.4:
                        malignant_labels[i] = 0
                elif i < num_samples * 0.85:
                    demographic = demographics[3]  # Hispanic
                else:
                    demographic = demographics[4]  # Other
                
                demographic_assignments.append(demographic)
            
            # Create splits
            splits = []
            for i in range(num_samples):
                if i < num_samples * 0.7:
                    splits.append("train")
                elif i < num_samples * 0.85:
                    splits.append("val")
                else:
                    splits.append("test")
            
            # Create DataFrame
            metadata = pd.DataFrame({
                "path": paths,
                "malignant": malignant_labels,
                "demographic": demographic_assignments,
                "split": splits,
            })
            
            # Save metadata
            metadata.to_csv(metadata_path, index=False)
            
            print(f"Created simulated metadata: {metadata_path}")
        else:
            print(f"Metadata already exists: {metadata_path}")
    
    # Split dataset
    if args.split:
        print("Splitting Pathology dataset...")
        
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        if os.path.exists(metadata_path):
            # Load metadata
            metadata = pd.read_csv(metadata_path)
            
            # Check if split column exists
            if "split" not in metadata.columns:
                print("Creating train/val/test splits...")
                
                # Set random seed
                random.seed(args.seed)
                np.random.seed(args.seed)
                
                # Create splits
                num_samples = len(metadata)
                indices = np.random.permutation(num_samples)
                
                train_end = int(num_samples * 0.7)
                val_end = int(num_samples * 0.85)
                
                train_indices = indices[:train_end]
                val_indices = indices[train_end:val_end]
                test_indices = indices[val_end:]
                
                # Assign splits
                splits = [""] * num_samples
                for i in train_indices:
                    splits[i] = "train"
                for i in val_indices:
                    splits[i] = "val"
                for i in test_indices:
                    splits[i] = "test"
                
                # Add split column
                metadata["split"] = splits
                
                # Save metadata
                metadata.to_csv(metadata_path, index=False)
                
                print("Splits created and saved.")
            else:
                print("Splits already exist in metadata.")
        else:
            print(f"Metadata file not found: {metadata_path}")
    
    print("Pathology dataset preparation complete.")


def prepare_retinal_dataset(args):
    """Prepare the Retinal dataset."""
    dataset_dir = os.path.join(args.output_dir, "retinal")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download dataset
    if args.download:
        print("Downloading Retinal dataset...")
        # Note: In a real implementation, you would download from the actual source
        # For this example, we'll simulate the download
        print("Retinal dataset download simulation complete.")
    
    # Preprocess dataset
    if args.preprocess:
        print("Preprocessing Retinal dataset...")
        
        # Create metadata
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            # In a real implementation, you would process actual image files
            # For this example, we'll create a simulated metadata file
            
            # Create simulated metadata
            num_samples = 1200
            demographics = ["African_American", "Asian", "Caucasian", "Hispanic", "Other"]
            
            # Create paths and labels
            paths = [f"images/retina_{i:04d}.png" for i in range(num_samples)]
            
            # DR grades: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
            dr_grades = np.random.randint(0, 5, size=num_samples)
            
            # Assign demographics with bias
            # Simulate a bias where certain demographics have higher rates of severe DR
            demographic_assignments = []
            for i in range(num_samples):
                if i < num_samples * 0.2:
                    demographic = demographics[0]  # African_American
                    # Increase severe DR rate for this demographic to simulate bias
                    if dr_grades[i] < 3 and random.random() < 0.3:
                        dr_grades[i] = random.randint(3, 4)
                elif i < num_samples * 0.4:
                    demographic = demographics[1]  # Asian
                elif i < num_samples * 0.7:
                    demographic = demographics[2]  # Caucasian
                    # Decrease severe DR rate for this demographic to simulate bias
                    if dr_grades[i] >= 3 and random.random() < 0.4:
                        dr_grades[i] = random.randint(0, 2)
                elif i < num_samples * 0.9:
                    demographic = demographics[3]  # Hispanic
                else:
                    demographic = demographics[4]  # Other
                
                demographic_assignments.append(demographic)
            
            # Create splits
            splits = []
            for i in range(num_samples):
                if i < num_samples * 0.7:
                    splits.append("train")
                elif i < num_samples * 0.85:
                    splits.append("val")
                else:
                    splits.append("test")
            
            # Create DataFrame
            metadata = pd.DataFrame({
                "path": paths,
                "diabetic_retinopathy_grade": dr_grades,
                "demographic": demographic_assignments,
                "split": splits,
            })
            
            # Save metadata
            metadata.to_csv(metadata_path, index=False)
            
            print(f"Created simulated metadata: {metadata_path}")
        else:
            print(f"Metadata already exists: {metadata_path}")
    
    # Split dataset
    if args.split:
        print("Splitting Retinal dataset...")
        
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        if os.path.exists(metadata_path):
            # Load metadata
            metadata = pd.read_csv(metadata_path)
            
            # Check if split column exists
            if "split" not in metadata.columns:
                print("Creating train/val/test splits...")
                
                # Set random seed
                random.seed(args.seed)
                np.random.seed(args.seed)
                
                # Create splits
                num_samples = len(metadata)
                indices = np.random.permutation(num_samples)
                
                train_end = int(num_samples * 0.7)
                val_end = int(num_samples * 0.85)
                
                train_indices = indices[:train_end]
                val_indices = indices[train_end:val_end]
                test_indices = indices[val_end:]
                
                # Assign splits
                splits = [""] * num_samples
                for i in train_indices:
                    splits[i] = "train"
                for i in val_indices:
                    splits[i] = "val"
                for i in test_indices:
                    splits[i] = "test"
                
                # Add split column
                metadata["split"] = splits
                
                # Save metadata
                metadata.to_csv(metadata_path, index=False)
                
                print("Splits created and saved.")
            else:
                print("Splits already exist in metadata.")
        else:
            print(f"Metadata file not found: {metadata_path}")
    
    print("Retinal dataset preparation complete.")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare datasets
    if args.dataset == "chexray" or args.dataset == "all":
        prepare_chexray_dataset(args)
    
    if args.dataset == "pathology" or args.dataset == "all":
        prepare_pathology_dataset(args)
    
    if args.dataset == "retinal" or args.dataset == "all":
        prepare_retinal_dataset(args)
    
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
