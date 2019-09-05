import os
import tempfile
import shutil


def make_simple_config_text(system_and_summary_paths):
    lines = []
    for system_path, summary_paths in system_and_summary_paths:
        line = "{} {}".format(system_path, " ".join(summary_paths))
        lines.append(line)
    return "\n".join(lines)

class TempFileManager(object):
    def __init__(self):
        pass

    def create_temp_files(self, texts):
        paths = []
        for text in texts:
            with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, dir=self.tmpdir) as fp:
                fp.write(text)
                paths.append(fp.name)
        return paths

    def create_temp_file(self, text):
        with tempfile.NamedTemporaryFile(
                mode="w", delete=False, dir=self.tmpdir) as fp:
            fp.write(text)
            return fp.name

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        return self

    def __exit__(self, *args):
        shutil.rmtree(self.tmpdir)
