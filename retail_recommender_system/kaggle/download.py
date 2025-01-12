# !pip install pydrive2[fsspec]

from pathlib import Path

from google.colab import auth
from oauth2client.client import GoogleCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.fs import GDriveFileSystem
from tqdm import tqdm


def _replace_remote_prefix(path: Path, remote_prefix: str, local_prefix: str) -> Path:
    return Path(local_prefix + str(path).replace(remote_prefix, ""))


def download_directory(fs, remote_prefix, local_prefix):
    files = [Path(f) for f in fs.glob(remote_prefix + "/**") if f.endswith(".py")]

    for file in tqdm(files):
        local_file = _replace_remote_prefix(file, remote_prefix, local_prefix)
        local_dir = _replace_remote_prefix(file.parents[0], remote_prefix, local_prefix)
        local_dir.mkdir(parents=True, exist_ok=True)

        with fs.open(file, "rb") as rf:
            with open(local_file, "wb") as lf:
                lf.write(rf.read())


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
fs = GDriveFileSystem("root", google_auth=gauth)

remote_prefix = "root/Projects"
local_prefix = "/kaggle/working"
download_directory(fs, remote_prefix, local_prefix)
