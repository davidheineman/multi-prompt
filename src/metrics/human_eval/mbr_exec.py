# Copyright (c) Meta Platforms, Inc. and affiliates.
# Taken from: https://github.com/facebookresearch/coder_reviewer_reranking/blob/main/execution.py

import pickle, subprocess, os, threading, signal, regex, tempfile

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
        return self.process.returncode


class PythonFunctionExecutor(object):
    def __init__(self, function_content, function_call, timeout=10):
        self.function_content = function_content
        self.function_content = self.function_content.replace("</code>", "")
        self.function_call = function_call
        self.timeout = timeout

    def __call__(self, i, use_json=False):
        tempdir = tempfile.TemporaryDirectory()
        with open(f"{tempdir.name}/code-{i}.py", "w", encoding='utf-8') as fout:
            print(self.function_content, file=fout)
            print(f"result = {self.function_call}", file=fout)
            print(f"import pickle", file=fout)
            print(
                f'pickle.dump(result, open("{tempdir.name}/execution_result-{i}.pkl", "wb"))',
                file=fout,
            )
        command = Command(f"python {tempdir.name}/code-{i}.py >/dev/null 2>&1")
        execution_status = command.run(timeout=self.timeout)
        if execution_status == 0:
            try:
                execution_results = pickle.load(
                    open(f"{tempdir.name}/execution_result-{i}.pkl", "rb")
                )
            except:
                execution_results = None
        else:
            execution_results = None
        tempdir.cleanup()
        return execution_status, execution_results
    

def humaneval_postprocess(completion):
    keep_lines = []
    for l in completion.split("\n"):
        if not l.startswith("print"):
            keep_lines.append(l)
    return "\n".join(keep_lines)


def humaneval_execute_one_assertion(prompt, completion, task_id, assertion):
    try:
        command = regex.match(f"assert (.+)==.+", assertion).group(1)
    except:
        command = regex.match(f"assert (.+)", assertion).group(1)
    python_function = prompt + completion
    executor = PythonFunctionExecutor(python_function, command)
    execution_result = executor(task_id.split("/")[1])
    return execution_result


def humaneval_execute_multiple_assertion(prompt, completion, task_id, assertions):
    execution_result = list()
    python_function = prompt + completion
    task_id = task_id.split("/")[1]
    for assertion_i, assertion in enumerate(assertions):
        try:
            try:
                command = regex.match(f"assert (.+)==.+", assertion).group(1)
            except:
                command = regex.match(f"assert (.+)", assertion).group(1)
        except:
            print(assertions)
            print(task_id)
            breakpoint()
        executor = PythonFunctionExecutor(python_function, command)
        execution_result.append(executor(f"{task_id}-{assertion_i}"))
    return execution_result


def humaneval_execute_generated_assertion(problem):
    execution_result = list()
    python_function = problem["prompt"] + problem["completion"]
    task_id = problem["task_id"].split("/")[1]

    total_matched = 0
    for assertion_i, assertion in enumerate(problem["gen_assertion"]):
        matched = False
        for pattern in ["assert (.+)==.+", "assert (.+) is .+", "assert (.+)"]:
            try:
                command = regex.match(pattern, assertion).group(1)
                matched = True
                break
            except:
                pass

        if matched:
            executor = PythonFunctionExecutor(python_function, command)
            execution_result.append(executor(f"{task_id}-{assertion_i}"))
            total_matched += int(matched)

        if total_matched > 20:
            break
    return execution_result
