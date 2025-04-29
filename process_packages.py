#!/usr/bin/env python3
import os

# read /run/secrets/hf_token in to HF_TOKEN environment variable
os.environ['HF_TOKEN'] = open('/run/secrets/hf_token').read().strip()

import re
import tarfile
import tempfile
import glob
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from itertools import batched
import semver
from tqdm import tqdm
from io import BytesIO, StringIO
from datasets import Dataset
from huggingface_hub import create_repo, upload_file

# Set up logger
logger = logging.getLogger('package_analyzer')
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

def process_file_in_tbz(tbz_path, processor_func, file_extensions=('ml', 'mli', 'dune', 'h', 'c', 'opam')):
    """
    Process files in a tbz archive without storing all contents in memory.
    
    Args:
        tbz_path: Path to the tbz file
        processor_func: Function that processes each file (gets file_type, file_path, content arguments)
        file_extensions: Tuple of file extensions to look for
    """
    try:
        # Open the tbz file
        with tarfile.open(tbz_path, 'r:bz2') as tar:
            # Iterate through the members (files) in the archive
            for member in tar.getmembers():
                # Skip directories
                if not member.isfile():
                    continue
                    
                # Get file extension
                _, ext = os.path.splitext(member.name)
                ext = ext.lstrip('.')
                
                # Special case for dune files (they might not have extension)
                if os.path.basename(member.name) == 'dune':
                    ext = 'dune'
                
                # Check if the file has one of our target extensions
                if ext in file_extensions:
                    try:
                        # Extract file contents
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            content = file_obj.read()
                            # Try to decode as text
                            try:
                                content = content.decode('utf-8')
                            except UnicodeDecodeError:
                                # If it fails, store as binary
                                content = str(content)
                            
                            # Process the file and then let it be garbage collected
                            processor_func(ext, member.name, content)
                    except Exception as e:
                        logger.error(f"Error extracting {member.name} from {tbz_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error reading {tbz_path}: {str(e)}")

def extract_metadata_from_tbz(tbz_path, package_name):
    """
    Extract metadata (license, homepage, dev-repo) from opam files in a tbz file.
    
    Args:
        tbz_path: Path to the tbz file
        package_name: Name of the package to look for specific opam files
        
    Returns:
        Dictionary with metadata information
    """
    metadata = {
        "license": "Unknown",
        "homepage": "Unknown",
        "dev_repo": "Unknown"
    }
    
    def process_opam_file(file_type, file_path, content):
        if file_type == 'opam':
            # Check if the opam file is for this package
            base_name = os.path.basename(file_path)
            if base_name == f"{package_name}.opam" or base_name == "opam":
                extracted_metadata = extract_metadata_from_opam(content)
                # Update metadata if values were found
                for key, value in extracted_metadata.items():
                    if value != "Unknown":
                        metadata[key] = value
    
    # Only look at opam files to save memory
    process_file_in_tbz(tbz_path, process_opam_file, file_extensions=('opam',))
    
    return metadata

def extract_license_from_tbz(tbz_path, package_name):
    """
    Extract license information from opam files in a tbz file (legacy function).
    
    Args:
        tbz_path: Path to the tbz file
        package_name: Name of the package to look for specific opam files
        
    Returns:
        String with license information or "Unknown" if not found
    """
    metadata = extract_metadata_from_tbz(tbz_path, package_name)
    return metadata["license"]

def list_tbz_file_contents(tbz_path, file_extensions=('ml', 'mli', 'dune', 'h', 'c', 'opam')):
    """
    List all files with specified extensions in a tbz file without extracting them all.
    
    Args:
        tbz_path: Path to the tbz file
        file_extensions: Tuple of file extensions to look for
    
    Returns:
        Dictionary with file extensions as keys and lists of matching files as values
    """
    result = defaultdict(list)
    
    try:
        # Open the tbz file
        with tarfile.open(tbz_path, 'r:bz2') as tar:
            # Iterate through the members (files) in the archive
            for member in tar.getmembers():
                # Skip directories
                if not member.isfile():
                    continue
                    
                # Get file extension
                _, ext = os.path.splitext(member.name)
                ext = ext.lstrip('.')
                
                # Special case for dune files (they might not have extension)
                if os.path.basename(member.name) == 'dune':
                    result['dune'].append(member.name)
                    continue
                
                # Check if the file has one of our target extensions
                if ext in file_extensions:
                    result[ext].append(member.name)
    
    except Exception as e:
        logger.error(f"Error reading {tbz_path}: {str(e)}")
    
    return result

def find_package_tbz_files(package_dir):
    """
    Find all tbz files in the given package directory
    """
    tbz_files = []
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.tbz'):
                tbz_files.append(os.path.join(root, file))
    return tbz_files

def extract_metadata_from_opam(opam_content):
    """
    Extract metadata information from opam file content
    
    Args:
        opam_content: String content of the opam file
    
    Returns:
        Dictionary with metadata information (license, homepage, dev-repo)
    """
    metadata = {
        "license": "Unknown",
        "homepage": "Unknown",
        "dev_repo": "Unknown"
    }
    
    # Extract license
    license_match = re.search(r'license:\s*"([^"]+)"', opam_content)
    if license_match:
        metadata["license"] = license_match.group(1)
    
    # Extract homepage
    homepage_match = re.search(r'homepage:\s*"([^"]+)"', opam_content)
    if homepage_match:
        metadata["homepage"] = homepage_match.group(1)
    
    # Extract dev-repo (git repository)
    # There are several possible formats for dev-repo in opam files
    repo_match = re.search(r'dev-repo:\s*"([^"]+)"', opam_content)
    if repo_match:
        metadata["dev_repo"] = repo_match.group(1)
    else:
        # Try alternative format
        repo_match = re.search(r'dev-repo:\s*([^\s]+)', opam_content)
        if repo_match:
            metadata["dev_repo"] = repo_match.group(1)
    
    return metadata

def extract_license_from_opam(opam_content):
    """
    Extract license information from opam file content (legacy function)
    
    Args:
        opam_content: String content of the opam file
    
    Returns:
        String with license information or "Unknown" if not found
    """
    metadata = extract_metadata_from_opam(opam_content)
    return metadata["license"]

def create_parquet_file_in_chunks(output_path, get_data_generator, schema=None, chunk_size=1000):
    """
    Create a parquet file by writing chunks of data to avoid using too much memory
    
    Args:
        output_path: Path to the output parquet file
        get_data_generator: Function that returns a generator of data rows
        schema: Optional pyarrow schema for the data
        chunk_size: Number of rows per chunk
    """
    import os
    import uuid
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write each chunk to a separate temporary parquet file, then combine them
    temp_files = []
    total_rows = 0
    
    try:
        # Process each chunk separately
        for chunk_num, chunk_data in enumerate(get_data_generator(chunk_size)):
            if not chunk_data:
                continue
                
            # Convert chunk to DataFrame
            df_chunk = pd.DataFrame(chunk_data)
            chunk_rows = len(df_chunk)
            total_rows += chunk_rows
            
            # Write this chunk to a separate temporary file
            temp_file = f"{output_path}.part_{chunk_num}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Writing chunk {chunk_num+1} with {chunk_rows} rows to temp file (total: {total_rows})")
            df_chunk.to_parquet(temp_file, engine='pyarrow', compression='zstd')
            temp_files.append(temp_file)
            
            # Clear memory
            del df_chunk
        
        if temp_files:
            # Now read and combine all temp parquet files
            logger.info(f"Combining {len(temp_files)} temporary parquet files...")
            
            # Create a schema from the first file to ensure consistency
            schema = pq.read_schema(temp_files[0])
            
            # Create the parquet writer with the schema
            with pq.ParquetWriter(output_path, schema) as writer:
                for temp_file in temp_files:
                    # Read each temp file and write its row groups to the final file
                    reader = pq.ParquetFile(temp_file)
                    for i in range(reader.num_row_groups):
                        row_group = reader.read_row_group(i)
                        writer.write_table(row_group)
            
            # Clean up temp files
            logger.info("Cleaning up temporary files...")
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")
        else:
            logger.warning("No data chunks collected to write to parquet file")
            return False, 0
    
    except Exception as e:
        logger.error(f"Error writing parquet file: {str(e)}")
        return False, 0
    
    return True, total_rows

def list_packages():
    """
    List all packages in the /cache/ directory, find the highest version of each,
    and analyze their tbz files for ml/mli/dune/.h/.c and opam files.
    """
    if not os.path.exists('/cache/'):
        logger.warning("Warning: /cache/ directory not found!")
        return [], {}, None
    
    # Dictionary to track highest version of each package
    package_versions = defaultdict(list)
    package_paths = {}  # Store full path to package directories
    
    # Walk through the cache directory
    for root, dirs, files in os.walk('/cache/'):
        for dir_name in dirs:
            # Check if directory name matches a versioned package pattern
            match = re.search(r'(.+?)[._-](\d+\.\d+\.\d+.*)', dir_name)
            if match:
                package_name = match.group(1)
                version_str = match.group(2)
                full_path = os.path.join(root, dir_name)
                
                # Normalize version string to make it compatible with semver
                # Remove any trailing components that aren't valid semver
                version_parts = version_str.split('.')
                if len(version_parts) >= 3:
                    # Extract the core version x.y.z
                    core_version = '.'.join(version_parts[:3])
                    # Add any pre-release or build metadata if present
                    if len(version_parts) > 3:
                        extra = '.'.join(version_parts[3:])
                        if '-' in extra or '+' in extra:
                            core_version += extra
                    
                    try:
                        # Try to parse as semver
                        semver.parse(core_version)
                        package_versions[package_name].append((version_str, core_version, full_path))
                    except ValueError:
                        # If parsing fails, just store the original version string
                        package_versions[package_name].append((version_str, '0.0.0', full_path))
    
    # Find highest version for each package
    unique_packages = {}
    package_paths = {}
    
    for package, versions in package_versions.items():
        if not versions:
            continue
            
        # Sort by semantic version
        sorted_versions = sorted(versions, key=lambda x: semver.VersionInfo.parse(x[1]) 
                               if semver.VersionInfo.is_valid(x[1]) else semver.VersionInfo.parse('0.0.0'))
        
        # Get the highest version
        highest_version = sorted_versions[-1][0]  # Original version string
        highest_version_path = sorted_versions[-1][2]  # Path to the highest version
        
        unique_packages[package] = highest_version
        package_paths[package] = highest_version_path
    
    # Now examine the tbz files for each unique package with highest version
    package_tbz_contents = {}
    
    # Create a generator function for parquet data
    def get_package_files_generator(chunk_size=1000):      
        for package_batch in batched(tqdm(package_paths.items()), chunk_size):
            for package, path in package_batch:
                chunk_data = []

                logger.info(f"Processing package: {package}")
                # Find all tbz files in this package directory
                tbz_files = find_package_tbz_files(path)
                version = unique_packages[package]
                
                if tbz_files:
                    package_tbz_contents[package] = {}
                    
                    # First pass: just get file listings for display and extract license
                    for tbz_file in tbz_files:
                        tbz_filename = os.path.basename(tbz_file)
                        logger.info(f"Processing tbz file: {tbz_filename}")
                        
                        # Get file listing for display
                        file_contents_listing = list_tbz_file_contents(tbz_file)
                        package_tbz_contents[package][tbz_filename] = file_contents_listing
                    
                    # Extract metadata information from first tbz file (to save memory)
                    package_metadata = {
                        "license": "Unknown",
                        "homepage": "Unknown",
                        "dev_repo": "Unknown"
                    }
                    
                    if tbz_files:
                        package_metadata = extract_metadata_from_tbz(tbz_files[0], package)
                        logger.info(f"Found metadata for {package}: license={package_metadata['license']}, " 
                                f"homepage={package_metadata['homepage']}, git repo={package_metadata['dev_repo']}")
                    
                    # Second pass: process files one by one to build chunk data
                    for tbz_file in tbz_files:
                        tbz_filename = os.path.basename(tbz_file)
                        logger.info(f"Extracting contents from: {tbz_filename}")
                        
                        # Process each file in the tbz file and add to current chunk
                        def process_file_for_parquet(file_type, file_path, content):
                            nonlocal chunk_data
                            
                            chunk_data.append({
                                'package_name': package,
                                'version': version,
                                'license': package_metadata['license'],
                                'homepage': package_metadata['homepage'],
                                'dev_repo': package_metadata['dev_repo'],
                                'file_type': file_type,
                                'file_path': file_path,
                                'file_contents': content
                            })
                        
                        # Process the tbz file
                        process_file_in_tbz(tbz_file, process_file_for_parquet)
                        
                    yield from chunk_data
    
    return unique_packages, package_tbz_contents, get_package_files_generator

def main():
    logger.info("Scanning /cache/ directory for packages...")
    unique_packages, package_tbz_contents, get_data_generator = list_packages()
    
    if not unique_packages:
        logger.warning("No packages found in /cache/ directory.")
        return
    
    logger.info(f"Found {len(unique_packages)} unique packages with their highest versions:")

    if get_data_generator:
        ds = Dataset.from_generator(get_data_generator)

        logger.info(f"Dataset created with {len(ds)} rows.")

        destination_dataset = "sadiqj/opam-archive-dataset"

        ds.push_to_hub(destination_dataset)
    else:
        logger.warning("No data generator available for Parquet file")

if __name__ == "__main__":
    main()
