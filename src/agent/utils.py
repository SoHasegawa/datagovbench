import contextlib
import io
import json
import traceback


def json_read(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def json_save(json_file, contents):
    with open(json_file, "w") as f:
        json.dump(contents, f, indent=2)


def execute_code(code):
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):
        with contextlib.redirect_stderr(error_buffer):
            try:
                exec(code, {})
            except Exception:
                traceback.print_exc()

    return output_buffer.getvalue(), error_buffer.getvalue()
