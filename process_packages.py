#!/usr/bin/env python3
import os
import re
import tarfile
import zipfile # Added for zip file support
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
logger.setLevel(logging.INFO) # Changed default to INFO, can be set by arg
# Create console handler if not already present (e.g. for multiple calls)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    # Set console_handler level via argument or default to INFO
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

# Matches package names and versions from directory names like:
# package-name.1.2.3 or package.1.2 or my.package-name.v0.1.0
PKG_VERSION_RE = re.compile(
    r"^(?P<name>[a-zA-Z0-9\._\-]+?)\."  # Package name (non-greedy to handle dots in name)
    r"(?P<v_prefix>v)?"  # Optional 'v' prefix
    r"(?P<major>[0-9]+)"  # Major version
    r"(?:\.(?P<minor>[0-9]+))?"  # Optional minor version
    r"(?:\.(?P<patch>[0-9]+))?"  # Optional patch version
    r"(?P<suffix>[~\+\-][a-zA-Z0-9\-\._]*)?$"  # Optional suffix (pre-release, build)
)


def setup_logging(level=logging.INFO):
    # Ensure logger is configured only once if this function is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else: # If handlers exist, just update their level
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

def process_archive_file(archive_path, processor_func, file_extensions=('ml', 'mli', 'dune', 'h', 'c', 'opam')):
    """
    Process files in a tar (tbz, tgz, tar.gz, tar.bz2) or zip archive 
    without storing all contents in memory.
    For tar files, uses 'r:*' to auto-detect compression.
    """
    archive_type = None
    if archive_path.endswith('.zip'):
        archive_type = 'zip'
    elif archive_path.endswith(('.tar.gz', '.tgz', '.tar.bz2', '.tbz')):
        archive_type = 'tar'
    else:
        logger.error(f"Unsupported archive format for {archive_path}")
        return

    if archive_type == 'zip':
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for member_info in zf.infolist():
                    if member_info.is_dir(): # Skip directories
                        continue
                    
                    file_path_in_archive = member_info.filename
                    _, ext = os.path.splitext(file_path_in_archive)
                    ext = ext.lstrip('.')
                    
                    if os.path.basename(file_path_in_archive) == 'dune':
                        ext = 'dune'
                    
                    if ext in file_extensions:
                        try:
                            content = zf.read(member_info.filename)
                            try:
                                content = content.decode('utf-8')
                            except UnicodeDecodeError:
                                content = str(content) # Store as string representation of bytes
                            processor_func(ext, file_path_in_archive, content)
                        except Exception as e:
                            logger.error(f"Error extracting {file_path_in_archive} from ZIP {archive_path}: {str(e)}")
        except zipfile.BadZipFile as e:
            logger.error(f"Error reading ZIP archive {archive_path} (bad ZIP file): {str(e)}")
        except Exception as e:
            logger.error(f"Generic error processing ZIP archive {archive_path}: {str(e)}")

    elif archive_type == 'tar':
        try:
            with tarfile.open(archive_path, 'r:*') as tar: # Auto-detect compression
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    
                    file_path_in_archive = member.name
                    _, ext = os.path.splitext(file_path_in_archive)
                    ext = ext.lstrip('.')
                    
                    if os.path.basename(file_path_in_archive) == 'dune':
                        ext = 'dune'
                    
                    if ext in file_extensions:
                        try:
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                content = file_obj.read()
                                try:
                                    content = content.decode('utf-8')
                                except UnicodeDecodeError:
                                    content = str(content) # Store as string representation of bytes
                                processor_func(ext, file_path_in_archive, content)
                        except Exception as e:
                            logger.error(f"Error extracting {file_path_in_archive} from TAR {archive_path}: {str(e)}")
        except tarfile.ReadError as e:
            logger.error(f"Error reading TAR archive {archive_path} (possibly corrupted or wrong format): {str(e)}")
        except Exception as e:
            logger.error(f"Generic error processing TAR archive {archive_path}: {str(e)}")

def extract_metadata_from_archive(archive_path, package_base_name):
    """
    Extract metadata (license, homepage, dev-repo) from opam files in an archive.
    Uses 'r:*' to auto-detect compression.
    """
    metadata = {
        "license": "Unknown",
        "homepage": "Unknown",
        "dev_repo": "Unknown"
    }
    
    # Regex to find opam files: either 'opam' or 'packagename.opam'
    # package_base_name should be the part before the version.
    opam_file_pattern = re.compile(rf"^(?:{re.escape(package_base_name)}\.opam|opam)$")

    def process_opam_file(file_type, file_path, content):
        # Check if the filename (basename) matches 'opam' or 'package_name.opam'
        if file_type == 'opam' and opam_file_pattern.match(os.path.basename(file_path)):
            logger.info(f"Processing opam file: {file_path} in archive {archive_path}") # Changed to info
            # Simple line-by-line parsing for opam fields
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("license:") or line.startswith("license :") : # handles 'license:' and 'license :'
                    metadata["license"] = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("homepage:") or line.startswith("homepage :"):
                    metadata["homepage"] = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("dev-repo:") or line.startswith("dev-repo :"):
                    metadata["dev_repo"] = line.split(":", 1)[1].strip().strip('"')
            logger.info(f"Metadata from {file_path}: {metadata}") # Changed to info

    # Use process_archive_file to find and process opam files
    # We only care about 'opam' extension here for metadata extraction.
    process_archive_file(archive_path, process_opam_file, file_extensions=('opam',))
    return metadata

def get_packages_from_cache(cache_path):
    packages = {}
    if not os.path.isdir(cache_path):
        logger.error(f"Cache directory {cache_path} not found.")
        return packages

    logger.info(f"Scanning cache directory: {cache_path}")
    dir_count = 0
    matched_count = 0
    for dirname in os.listdir(cache_path):
        dir_count += 1
        m = PKG_VERSION_RE.match(dirname)
        if m:
            matched_count +=1
            name = m.group("name")
            if not name: # Prevent empty package names
                logger.warning(f"Directory '{dirname}' matched pattern but resulted in an empty package name. Skipping.")
                continue

            major = m.group("major")
            minor = m.group("minor") or "0"
            patch = m.group("patch") or "0"
            suffix = m.group("suffix") or ""
            
            version_str_for_parse = f"{major}.{minor}.{patch}{suffix}"
            if suffix.startswith("~"):
                version_str_for_parse = f"{major}.{minor}.{patch}-{suffix[1:]}"
            
            logger.info(f"Matched directory: '{dirname}'. Name: '{name}', Raw Version parts: M={major},m={minor},p={patch},suf='{suffix}'. Semver attempt: '{version_str_for_parse}'") # Changed to info

            try:
                version = semver.VersionInfo.parse(version_str_for_parse)
                if name not in packages or version > packages[name]["version_info"]:
                    packages[name] = {
                        "version_info": version,
                        "version_str": str(version), # Store the normalized semver string
                        "dir": dirname # Original directory name, e.g., gstreamer.0.3.0
                    }
                    logger.info(f"Accepted/Updated package: {name} with version {str(version)} from dir {dirname}") # Changed to info
            except ValueError:
                logger.warning(f"Could not parse version '{version_str_for_parse}' for package '{name}' from dir '{dirname}'. Skipping this directory.")
        # else:
            # logger.debug(f"Directory '{dirname}' did not match package pattern.") # This can remain debug as it's very verbose
            
    logger.info(f"Scanned {dir_count} directories, {matched_count} matched package pattern.")
    logger.info(f"Found {len(packages)} unique highest package versions.")
    if packages:
        logger.info(f"First 5 packages found (name, info): {list(packages.items())[:5]}") # Changed to info
    return packages

def main(cache_path='/cache/', output_dir='/root/opam-archive-dataset/data', batch_size=1000, log_level_str="INFO"):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    setup_logging(log_level)

    logger.info(f"Starting package processing. Cache: {cache_path}, Output: {output_dir}, Batch: {batch_size}")
    
    packages_to_process = get_packages_from_cache(cache_path)
    
    if not packages_to_process:
        logger.warning("No packages found to process after scanning cache. Exiting.")
        return

    logger.info(f"Total unique package versions selected for processing: {len(packages_to_process)}")

    package_data = []
    processed_count = 0
    skipped_no_archive_count = 0
    skipped_empty_archive_count = 0

    for package_name, package_info in tqdm(packages_to_process.items(), desc="Processing packages"):
        # Log the package key and its associated info (which includes the directory name)
        logger.info(f"Loop iteration for package key: '{package_name}'. Associated info from cache scan: {package_info}")

        version_str = package_info["version_str"]
        package_dir_name = package_info["dir"] # This is the original directory name like 'gstreamer.0.3.0'
        
        logger.info(f"Starting processing for package: {package_name} (version: {version_str}, specific source dir in cache: {package_dir_name})")

        current_package_contents = []
        package_metadata = {
            "license": "Unknown", "homepage": "Unknown", "dev_repo": "Unknown",
            "processed_opam_file_for_metadata": False
        }

        package_version_full_path = os.path.join(cache_path, package_dir_name)
        
        # Log the contents of the package version directory
        if os.path.isdir(package_version_full_path):
            try:
                dir_contents = os.listdir(package_version_full_path)
                logger.info(f"Contents of directory {package_version_full_path}: {dir_contents}")
            except OSError as e:
                logger.error(f"Could not list contents of directory {package_version_full_path}: {e}")
        else:
            logger.warning(f"Package directory {package_version_full_path} not found or is not a directory when trying to list contents.")

        # Glob for .tbz, .tgz, .tar.gz, .tar.bz2 and .zip files
        archive_files = glob.glob(os.path.join(package_version_full_path, '*.tbz'))
        archive_files.extend(glob.glob(os.path.join(package_version_full_path, '*.tgz')))
        archive_files.extend(glob.glob(os.path.join(package_version_full_path, '*.tar.gz')))
        archive_files.extend(glob.glob(os.path.join(package_version_full_path, '*.tar.bz2')))
        archive_files.extend(glob.glob(os.path.join(package_version_full_path, '*.zip')))

        if not archive_files:
            logger.warning(f"No .tbz, .tgz, .tar.gz, .tar.bz2, or .zip archives found in {package_version_full_path} for package {package_name} {version_str}. This package version will be skipped.")
            package_data.append({
                "package_name": package_name, "version": version_str,
                "license": "Skipped", "homepage": "Skipped", "dev_repo": "Skipped",
                "files": [], "error": "No .tbz, .tgz, .tar.gz, .tar.bz2, or .zip archive found"
            })
            skipped_no_archive_count += 1
            continue

        logger.info(f"Found {len(archive_files)} archive(s) in {package_version_full_path}: {archive_files}")

        for archive_path in archive_files:
            logger.info(f"Processing archive file: {archive_path} for package {package_name} {version_str}")

            archive_name_for_meta_lookup = package_name # Default to package_name from directory
            archive_filename_base = os.path.basename(archive_path)
            # Try to parse archive filename for a more accurate metadata package name hint
            # Regex: (name_part)[-._v](version_part_digits...)
            # Captures the part before a version-like string in the archive filename.
            # e.g., "package-name-1.2.3.tar.gz" -> "package-name"
            # e.g., "h2-0.13.0.tbz" -> "h2"
            meta_name_match = re.match(r"^(.*?)(?:[._-](?:v\d|\d))", archive_filename_base)
            if meta_name_match:
                potential_name_from_archive = meta_name_match.group(1)
                if potential_name_from_archive and potential_name_from_archive != package_name:
                    logger.info(f"Archive filename '{archive_filename_base}' suggests base name '{potential_name_from_archive}' for metadata lookup, differing from dir-derived name '{package_name}'. Using '{potential_name_from_archive}' for opam file lookup within this archive.")
                    archive_name_for_meta_lookup = potential_name_from_archive
                elif not potential_name_from_archive:
                    logger.warning(f"Could not reliably determine base name from archive '{archive_filename_base}' for metadata. Defaulting to dir-derived name '{package_name}'.")

            if not package_metadata["processed_opam_file_for_metadata"]:
                current_archive_metadata = extract_metadata_from_archive(archive_path, archive_name_for_meta_lookup)
                if any(current_archive_metadata[k] != "Unknown" for k in ["license", "homepage", "dev_repo"]):
                    package_metadata.update(current_archive_metadata)
                    package_metadata["processed_opam_file_for_metadata"] = True
                    logger.info(f"Updated metadata (source: {archive_name_for_meta_lookup}.opam or opam in {archive_filename_base}): License='{package_metadata['license']}', Homepage='{package_metadata['homepage']}', DevRepo='{package_metadata['dev_repo']}'")
                else:
                    logger.info(f"No new metadata found in opam file (related to '{archive_name_for_meta_lookup}') within {archive_filename_base}")
            
            # Define a collector function for this archive's contents
            def content_collector(file_type, file_path_in_archive, content):
                current_package_contents.append({
                    "file_type": file_type,
                    "file_path": file_path_in_archive, # Path within the archive
                    "content": content
                })

            logger.info(f"Extracting contents from: {archive_path}") # Changed to info
            process_archive_file(archive_path, content_collector)
        
        # After processing all archives for this package version
        if not current_package_contents:
            logger.warning(f"No processable files (ml, mli, etc.) found in any archives for {package_name} {version_str} in {package_version_full_path}.")
            # Still add an entry, it might have metadata, or we need to record it was empty
            package_data.append({
                "package_name": package_name, "version": version_str,
                "license": package_metadata["license"], "homepage": package_metadata["homepage"], "dev_repo": package_metadata["dev_repo"],
                "files": [], "error": "Archives found but contained no processable files"
            })
            skipped_empty_archive_count +=1
        else:
            package_data.append({
                "package_name": package_name, "version": version_str,
                "license": package_metadata["license"], "homepage": package_metadata["homepage"], "dev_repo": package_metadata["dev_repo"],
                "files": current_package_contents
            })
            processed_count +=1
            logger.info(f"Finished processing for {package_name} {version_str}. Total files extracted: {len(current_package_contents)}")

    logger.info(f"Finished processing all packages. Successfully processed: {processed_count}, Skipped (no archive): {skipped_no_archive_count}, Skipped (empty/no processable files): {skipped_empty_archive_count}")

    # Convert to Hugging Face Dataset
    if not package_data:
        logger.warning("No data collected from packages. Skipping dataset creation and upload.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Convert list of dicts to a dict of lists for Dataset.from_dict
    # Handle potential missing 'error' key
    data_dict = defaultdict(list)
    for item in package_data:
        data_dict["package_name"].append(item.get("package_name"))
        data_dict["version"].append(item.get("version"))
        data_dict["license"].append(item.get("license"))
        data_dict["homepage"].append(item.get("homepage"))
        data_dict["dev_repo"].append(item.get("dev_repo"))
        data_dict["files"].append(item.get("files")) # files is a list of dicts
        data_dict["error"].append(item.get("error", None)) # Add error field, default to None

    try:
        dataset = Dataset.from_dict(data_dict)
        logger.info(f"Successfully created Hugging Face Dataset. Number of rows: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to create Hugging Face Dataset: {e}")
        # Fallback or detailed logging for why from_dict failed
        logger.error("Data structure that failed:")
        for key, value_list in data_dict.items():
            logger.error(f"  {key}: list of {len(value_list)} items. First item type: {type(value_list[0]) if value_list else 'N/A'}")
            if key == "files" and value_list and isinstance(value_list[0], list) and value_list[0]:
                 logger.error(f"    First item in 'files' list is a list of {len(value_list[0])} dicts. Keys of first dict: {value_list[0][0].keys() if value_list[0] else 'N/A'}")
        return


    # Save to Parquet in batches
    # Calculate number of batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    logger.info(f"Preparing to save dataset in {num_batches} Parquet batch(es) of size {batch_size}.")

    for i in range(num_batches):
        batch_dataset = dataset.shard(num_shards=num_batches, index=i)
        parquet_file_path = os.path.join(output_dir, f"opam_packages_batch_{i}.parquet")
        try:
            # Convert to pandas DataFrame first for more control or direct save if supported well
            # df = batch_dataset.to_pandas()
            # table = pa.Table.from_pandas(df)
            # pq.write_table(table, parquet_file_path)
            batch_dataset.to_parquet(parquet_file_path) # Simpler if it works
            logger.info(f"Successfully saved batch {i} to {parquet_file_path}")
        except Exception as e:
            logger.error(f"Error saving batch {i} to Parquet: {str(e)}")
            # Consider alternative saving or logging more details
            # logger.error(f"Data for batch {i}: {batch_dataset.to_dict()}")

    try:
        destination_dataset = "sadiqj/opam-archive-dataset"
        logger.info(f"Pushing dataset to Hugging Face Hub at {destination_dataset}")
        dataset.push_to_hub(destination_dataset)
        logger.info(f"Successfully pushed dataset to {destination_dataset}")
    except Exception as e:
        logger.error(f"Failed to push dataset to Hugging Face Hub: {e}")

    logger.info("Package processing and Parquet export complete.")


if __name__ == '__main__':
    # Example: allow setting log level from command line, defaulting to INFO
    import argparse
    parser = argparse.ArgumentParser(description="Process OCaml packages from archives.")
    parser.add_argument("--cache_path", default="/cache/", help="Path to the cache directory.")
    parser.add_argument("--output_dir", default="/root/opam-archive-dataset/data", help="Directory to save Parquet files.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of package files per Parquet batch.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    main(cache_path=args.cache_path, output_dir=args.output_dir, batch_size=args.batch_size, log_level_str=args.log_level)
