import zipfile

with zipfile.ZipFile("/tmp/model.zip") as zf:
    zf.extractall("/tmp/model")

import tarfile

with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
    tar.add("/tmp/model/bundle.json", arcname="bundle.json")
    tar.add("/tmp/model/root", arcname="root")