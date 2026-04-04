import gzip
import shutil
from pathlib import Path

def gunzip_file(source_path, dest_path=None):
    source= Path(source_path)
    if dest_path is None:
        dest_path= source.with_suffix('')
    with gzip.open(source, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted: {dest_path}")