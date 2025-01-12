import time
from pathlib import Path

from oauth2client.client import GoogleCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.fs import GDriveFileSystem
from tqdm import tqdm

gauth = GoogleAuth()
gauth.LocalWebserverAuth()


def delete_files(fs, remote_directory):
    if not fs.exists(str(remote_directory)):
        print(f"{remote_directory} does not exist.")
        return
    files = fs.ls(str(remote_directory))
    confirm = input(f"Do you really want to delete all files in {remote_directory}? \nFiles: \n {files} \n\nAnswer (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborting deletion.")
        return
    fs.rm(remote_directory)
    time.sleep(5)


def upload_files(fs, remote_directory, local_directory):
    files = [Path(f) for f in local_directory.rglob("*") if f.suffix == ".py"]

    for file in tqdm(files):
        with open(file, "rb") as f:
            with fs.open(remote_directory / file, "wb") as rf:
                rf.write(f.read())


def main():
    dirname = "retail_recommender_system"
    local_directory = Path(".") / dirname
    remote_directory = "root/Projects"

    fs = GDriveFileSystem("root", google_auth=gauth)

    delete_files(fs, Path(remote_directory) / dirname)
    upload_files(fs, remote_directory, local_directory)


if __name__ == "__main__":
    main()
