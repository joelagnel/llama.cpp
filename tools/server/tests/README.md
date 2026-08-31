# Server tests

Python based server tests scenario using [pytest](https://docs.pytest.org/en/stable/).

Tests target GitHub workflows job runners with 4 vCPU.

Note: If the host architecture inference speed is faster than GitHub runners one, parallel scenario may randomly fail.
To mitigate it, you can increase values in `n_predict`, `kv_size`.

### Install dependencies

`pip install -r requirements.txt`

### Run tests

1. Build the server

```shell
cd ../../..
cmake -B build
cmake --build build --target llama-server
```

2. Start the test: `./tests.sh`

It's possible to override some scenario steps values with environment variables:

| variable                 | description                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------|
| `PORT`                   | `context.server_port` to set the listening port of the server during scenario, default: `8080` |
| `LLAMA_SERVER_BIN_PATH`  | to change the server binary path, default: `../../../build/bin/llama-server`                         |
| `DEBUG`                  | to enable steps and server verbose mode `--verbose`                                       |
| `N_GPU_LAYERS`           | number of model layers to offload to VRAM `-ngl --n-gpu-layers`                                |
| `LLAMA_CACHE`            | by default server tests re-download models to the `tmp` subfolder. Set this to your cache (e.g. `$HOME/Library/Caches/llama.cpp` on Mac or `$HOME/.cache/llama.cpp` on Unix) to avoid this |

To run slow tests (will download many models, make sure to set `LLAMA_CACHE` if needed):

```shell
SLOW_TESTS=1 ./tests.sh
```

To run with stdout/stderr display in real time (verbose output, but useful for debugging):

```shell
DEBUG=1 ./tests.sh -s -v -x
```

To run all the tests in a file:

```shell
./tests.sh unit/test_chat_completion.py -v -x
```

To run a single test:

```shell
./tests.sh unit/test_chat_completion.py::test_invalid_chat_completion_req
```

### Focused LlamaScope telemetry checks

The private-fork telemetry contract has native MoE tests and authenticated server scenarios. Configure and build the server and native test targets first, then run the focused CTest selection:

```shell
cmake -S . -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server test-moe-routing test-moe-routing-telemetry
ctest --test-dir build --output-on-failure -R "^(test-moe-routing-telemetry|test-moe-routing-dense|test-moe-routing-dsv4)$"
```

Run the server scenarios from this directory. Set `LLAMA_SERVER_BIN_PATH` to the server from the same build tree when it is not the default path:

```shell
LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server \
  ./tests.sh unit/test_telemetry_control.py unit/test_telemetry.py -v -x
```

On Windows PowerShell, the equivalent environment setup is:

```powershell
$env:LLAMA_SERVER_BIN_PATH = (Resolve-Path ..\..\..\build\bin\llama-server.exe)
py -m pytest -q .\unit\test_telemetry_control.py .\unit\test_telemetry.py
```

These checks cover full-replacement `/props` control, authentication, actual-listener loopback enforcement, environment/request non-activation, restart reset, dense-path behavior, canonical schema-v2 bytes, sparse topology, chunk boundaries, loss/gaps, peer coverage, MTP linkage, and generic telemetry behavior.

The CUDA routing-readback gate is separate and Windows-only. It requires a static CUDA build and proves that enabled MoE capture performs device-to-host readback while disabled and dense paths do not. Configure it explicitly; the configuration rejects a CPU build, a shared build, or a non-Windows build:

```powershell
cmake -S . -B build-cuda-routing -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF `
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_MOE_ROUTING_CUDA_READBACK_TESTS=ON
cmake --build build-cuda-routing --target test-moe-routing
ctest --test-dir build-cuda-routing --output-on-failure `
  -R "^test-moe-routing-dsv4-cuda-readback$"
```

Hint: You can compile and run test in single command, useful for local development:

```shell
cmake --build build -j --target llama-server && ./tools/server/tests/tests.sh
```

To see all available arguments, please refer to [pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html)

### Debugging external llama-server
It can sometimes be useful to run the server in a debugger when invesigating test
failures. To do this, the environment variable `DEBUG_EXTERNAL=1` can be set
which will cause the test to skip starting a llama-server itself. Instead, the
server can be started in a debugger.

Example using `gdb`:
```console
$ gdb --args ../../../build/bin/llama-server \
    --host 127.0.0.1 --port 8080 \
    --temp 0.8 --seed 42 \
    --hf-repo ggml-org/models --hf-file tinyllamas/stories260K.gguf \
    --batch-size 32 --no-slots --alias tinyllama-2 --ctx-size 512 \
    --parallel 2 --n-predict 64
```
And a break point can be set in before running:
```console
(gdb) br server.cpp:4604
(gdb) r
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
```

And then the test in question can be run in another terminal:
```console
(venv) $ env DEBUG_EXTERNAL=1 ./tests.sh unit/test_chat_completion.py -v -x
```
And this should trigger the breakpoint and allow inspection of the server state
in the debugger terminal.
